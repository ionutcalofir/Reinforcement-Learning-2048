import os
from absl import flags
from absl import app
from absl import logging

import engine
import custom_utils

FLAGS = flags.FLAGS

flags.DEFINE_enum('phase', 'test', ['train', 'test'], 'Phase to run.')
flags.DEFINE_string('logdir', './logdir', 'Where to save the models.')
flags.DEFINE_string('model_name', 'dqn', 'Model name.')

def main(_):
    os.makedirs(FLAGS.logdir, exist_ok=True)
    logdir = custom_utils.create_dir(FLAGS.logdir, FLAGS.model_name)

    eng = engine.Engine(logdir=logdir,
                        phase=FLAGS.phase)

    if FLAGS.phase == 'train':
        eng.train()
    elif FLAGS.phase == 'test':
        model_path = '/home/ionutc/Documents/Repositories/Reinforcement-Learning-2048/2048/logdir/327-dqn/models/model_55944.pt'
        eng.test(model_path)
    else:
        raise Exception('Phase {} unknown!'.format(FLAGS.phase))

if __name__ == '__main__':
    app.run(main)
