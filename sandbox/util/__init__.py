import logging

from blocks.extensions import SimpleExtension
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.extensions.predicates import OnLogRecord


logger = logging.getLogger(__name__)


class CompositeExtension(SimpleExtension):
    """An extension that manages several other extensions.
    Parameters
    ----------
    sub_extensions : iterable
        An iterable collection of sub-extensions to manage.
    run_before_children : bool, optional
        Whether the container extension's own logic should
        be dispatched before that of the sub-extensions.
        If ``False``, the containing extension is dispatched last.
        Defaults to ``True``.
    Notes
    -----
    The main use case for this class is bundling together groups
    of extensions that are most commonly used in tandem, configured
    so as to interact with one another. Encapsulating this pattern
    in a single extension reduces boilerplate.
    Sub-extensions are dispatched in the order specified in
    ``sub_extensions``, on whatever triggers they are individually
    configured to respect.
    Sub-extensions may be run on different triggers than the containing
    extension; the trigger keywords passed to the constructor
    for this class only affect the outer extension's logic, and
    sub-extensions should be configured independently (possibly in
    a constructor for a subclass of :class:`CompositeExtension`).
    """
    def __init__(self, sub_extensions, run_before_children=True, **kwargs):
        self.sub_extensions = sub_extensions
        self.run_before_children = run_before_children
        super(CompositeExtension, self).__init__(**kwargs)

    def dispatch(self, callback_invoked, *from_main_loop):
        def run_super():
            super(CompositeExtension, self).dispatch(callback_invoked,
                                                     *from_main_loop)
        if self.run_before_children:
            run_super()

        for ext in self.sub_extensions:
            ext.dispatch(callback_invoked, *from_main_loop)

        if not self.run_before_children:
            run_super()

    @property
    def main_loop(self):
        return super(CompositeExtension, self).main_loop

    @main_loop.setter
    def main_loop(self, value):
        self._main_loop = value
        for sub in self.sub_extensions:
            sub.main_loop = value

    def do(self, which_callback, *args):
        pass


class EarlyStopping(CompositeExtension):
    """A 'batteries-included' early stopping extension.
    Parameters
    ----------
    record_name : str
        The log record entry whose value represents the quantity to base
        early stopping decisions on, e.g. some measure of validation set
        performance.
    checkpoint_extension : :class:`~blocks.extensions.Checkpoint`, optional
        A :class:`~blocks.extensions.Checkpoint` instance to configure to
        save a checkpoint when a new best performer is found.
    checkpoint_filename : str, optional
        The filename to use for the 'current best' checkpoint. Must be
        provided if ``checkpoint_extension`` is specified.
    notification_name : str, optional
        The name to be written in the log when a new best-performing
        model is found. Defaults to ``record_name + '_best_so_far'``.
    choose_best : callable, optional
        See :class:`TrackTheBest`.
    iterations : int, optional
        See :class:`FinishIfNoImprovementAfter`.
    epochs : int, optional
        See :class:`FinishIfNoImprovementAfter`.
    Notes
    -----
    .. warning::
        If you want the best model to be saved, you need to specify
        a value for the ``checkpoint_extension`` and
        ``checkpoint_filename`` arguments!
    Trigger keyword arguments will affect how often the log is inspected
    for the record name (in order to determine if a new best has been
    found), as well as how often a decision is made about whether to
    continue training. By default, ``after_epoch`` is set,
    as is ``before_training``, where some sanity checks are performed
    (including the optional self-management of checkpointing).
    If ``checkpoint_extension`` is not in the main loop's extensions list
    when the `before_training` trigger is run, it will be added as a
    sub-extension of this object.
    Examples
    --------
    To simply track the best value of a log entry and halt training
    when it hasn't improved in a sufficient amount of time, we could
    use e.g.
    >>> stopping_ext = EarlyStopping('valid_error', iterations=100)
    which would halt training if a new minimum ``valid_error`` has not
    been achieved in 100 iterations (i.e. minibatches/steps). To measure
    in terms of epochs (which usually correspond to passes through the
    training set), you could use
    >>> epoch_stop_ext = EarlyStopping('valid_error', epochs=5)
    If you are tracking a log entry where there's a different definition
    of 'best', you can provide a callable that takes two log values
    and returns the one that :class:`EarlyStopping` should consider
    "better". For example, if you were tracking accuracy, where higher
    is better, you could pass the built-in ``max`` function:
    >>> max_acc_stop = EarlyStopping('valid_accuracy', choose_best=max,
    ...                              notification_name='highest_acc',
    ...                              epochs=10)
    Above we've also provided an alternate notification name, meaning
    a value of ``True`` will be written under the entry name
    ``highest_acc`` whenever a new highest accuracy is found (by default
    this would be a name like ``valid_accuracy_best_so_far``).
    Let's configure a checkpointing extension to save the model and log
    (but not the main loop):
    >>> from blocks.extensions.saveload import Checkpoint
    >>> checkpoint = Checkpoint('my_model.tar', save_main_loop=False,
    ...                         save_separately=['model', 'log'],
    ...                         after_epoch=True)
    When we pass this object to :class:`EarlyStopping`, along with a
    different filename, :class:`EarlyStopping` will configure that same
    checkpointing extension to *also* serialize to ``best_model.tar`` when
    a new best value of validation error is achieved.
    >>> stopping = EarlyStopping('valid_error', checkpoint,
    ...                          'best_model.tar', epochs=5)
    Finally, we'll set up the main loop:
    >>> from blocks.main_loop import MainLoop
    >>> # You would, of course, use a real algorithm and data stream here.
    >>> algorithm = data_stream = None
    >>> main_loop = MainLoop(algorithm=algorithm,
    ...                      data_stream=data_stream,
    ...                      extensions=[stopping, checkpoint])
    Note that you do want to place the checkpoint extension *after*
    the stopping extension, so that the appropriate log records
    have been written in order to trigger the checkpointing
    extension.
    It's also possible to in-line the creation of the
    checkpointing extension:
    >>> main_loop = MainLoop(algorithm=algorithm,
    ...                      data_stream=data_stream,
    ...                      extensions=[EarlyStopping(
    ...                          'valid_error',
    ...                          Checkpoint('my_model.tar',
    ...                                     save_main_loop=False,
    ...                                     save_separately=['model',
    ...                                                      'log'],
    ...                                     after_epoch=True),
    ...                          'my_best_model.tar',
    ...                          epochs=5)])
    Note that we haven't added the checkpointing extension to the
    main loop's extensions list. No problem: :class:`EarlyStopping` will
    detect that it isn't being managed by the main loop and manage it
    internally. It will automatically be executed in the right order
    for it to function properly alongside :class:`EarlyStopping`.
    """
    def __init__(self, record_name, checkpoint_extension=None,
                 checkpoint_filename=None, notification_name=None,
                 choose_best=min, iterations=None, epochs=None, **kwargs):
        if notification_name is None:
            notification_name = record_name + '_best_so_far'
        kwargs.setdefault('after_epoch', True)
        tracking_ext = TrackTheBest(record_name, notification_name,
                                    choose_best=choose_best, **kwargs)
        stopping_ext = FinishIfNoImprovementAfter(notification_name,
                                                  iterations=iterations,
                                                  epochs=epochs,
                                                  **kwargs)
        self.checkpoint_extension = checkpoint_extension
        if checkpoint_extension and checkpoint_filename:
            checkpoint_extension.add_condition(['after_batch'],
                                               OnLogRecord(notification_name),
                                               (checkpoint_filename,))
        elif checkpoint_extension is not None and checkpoint_filename is None:
            raise ValueError('checkpoint_extension specified without '
                             'checkpoint_filename')
        kwargs.setdefault('before_training', True)
        super(EarlyStopping, self).__init__([tracking_ext, stopping_ext],
                                            **kwargs)

    def do(self, which_callback, *args):
        if which_callback == 'before_training' and self.checkpoint_extension:
            if self.checkpoint_extension not in self.main_loop.extensions:
                logger.info('%s: checkpoint extension %s not in main loop '
                            'extensions, adding as sub-extension of %s',
                            self.__class__.__name__, self.checkpoint_extension,
                            self)
                self.checkpoint_extension.main_loop = self.main_loop
                self.sub_extensions.append(self.checkpoint_extension)
            else:
                exts = self.main_loop.extensions
                if exts.index(self.checkpoint_extension) < exts.index(self):
                    logger.warn('%s: configured checkpointing extension '
                                'appears after this extension in main loop '
                                'extensions list. This may lead to '
                                'unwanted results, as the notification '
                                'that would trigger serialization '
                                'of a new best will not have been '
                                'written yet when the checkpointing '
                                'extension is run.', self.__class__.__name__)
