
from __future__ import division, print_function

import os
import shutil
import theano
import theano.tensor as T

from blocks.extensions.saveload import Checkpoint, SAVED_TO
from blocks.serialization import secure_dump
import cPickle
from sample_location import generate_samples


class SampleAttentionCheckpoint(Checkpoint):
    def __init__(self, monitor_img, image_size, channels,
                 save_subdir, **kwargs):
        super(SampleAttentionCheckpoint, self).__init__(path=None, **kwargs)
        self.image_size = image_size
        self.monitor_img = monitor_img
        self.channels = channels
        self.save_subdir = save_subdir
        self.iteration = 0
        self.epoch_src = "{0}/sample.png".format(save_subdir)

    def do(self, callback_name, *args):
        """Sample the model and save images to disk
        """
        generate_samples(self.main_loop.model, self.monitor_img,
                         self.save_subdir, self.image_size, self.channels)
        if os.path.exists(self.epoch_src):
            epoch_dst = "{0}/epoch-{1:03d}.png".format(self.save_subdir,
                                                       self.iteration)
            self.iteration = self.iteration + 1
            shutil.copy2(self.epoch_src, epoch_dst)
            os.system("convert -delay 5 -loop 1 "
                      "{0}/epoch-*.png {0}/training.gif".format(
                          self.save_subdir))


class PartsOnlyCheckpoint(Checkpoint):
    def do(self, callback_name, *args):
        """Pickle the save_separately parts (and not the main loop object) to disk.

        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.

        """
        _, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if from_user:
                path, = from_user
            ### this line is disabled from superclass impl to bypass using blocks.serialization.dump
            ### because pickling main thusly causes pickling error:
            ### "RuntimeError: maximum recursion depth exceeded while calling a Python object"
            # secure_dump(self.main_loop, path, use_cpickle=self.use_cpickle)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                secure_dump(getattr(self.main_loop, attribute),
                            filenames[attribute], cPickle.dump)
        except Exception:
            path = None
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (path,))
