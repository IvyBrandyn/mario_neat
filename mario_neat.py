import retro
import numpy as np
import cv2
import neat
import pickle
from numpy.lib.stride_tricks import as_strided
from PIL import Image


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        game_to_open = 'SuperMarioBros-Nes'
        level_to_play = 'Level1-1'
        self.retro_environment = retro.make(game_to_open, level_to_play)
        self.retro_environment.reset()

        game_frame = self.retro_environment.reset()
        image_width, image_height, _ = self.retro_environment.observation_space.shape

        new_frame_size = (image_width, image_height)

        feed_forward_network = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        generational_max_fitness = 0
        current_genome_fitness = 0
        frame = 0
        done = False

        while not done:
            # Render frame.  Translates to seeing the game.
            if current_genome_fitness > 0:
                self.retro_environment.render()

            # Remove color since it is not important
            greyscale = cv2.COLOR_BGR2GRAY
            game_frame = cv2.cvtColor(game_frame, greyscale)

            # Normalization
            # game_frame = normalize_image(game_frame)

            # Global centering
            # game_frame = global_centering(game_frame)

            # Max pooling
            game_frame = np.reshape(game_frame, new_frame_size)
            game_frame = pool2d(game_frame, kernel_size=8, stride=8, padding=0, pool_mode='max')

            game_frame = change_contrast(game_frame, 50)
            game_frame = np.asarray(game_frame)

            one_dimensional_image = np.ndarray.flatten(game_frame)

            # Give actions generated from neural network after considering frame
            actions_from_rnn = feed_forward_network.activate(one_dimensional_image)

            # See what happens when applying actions
            game_frame, reward, done, info = self.retro_environment.step(actions_from_rnn)

            current_genome_fitness += reward
            if current_genome_fitness > generational_max_fitness:
                generational_max_fitness = current_genome_fitness
                # Allow current leader in fitness to continue to explore potential
                frame = 0
            else:
                # Continue towards timeout
                frame += 1

            # Attempt training for 250 frames
            if done or frame == 250:
                done = True
                #if current_genome_fitness > 0:
                    #print(self.genome.__str__())
                self.retro_environment.render(close=True)
                # Update genome fitness
                return current_genome_fitness


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()


def change_contrast(img, level):
    img = Image.fromarray(img)
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


def normalize_image(img):
    img = img.astype('float32')
    img /= 255.0
    return img


def global_centering(img):
    img_mean = img.mean()
    img = img - img_mean
    return img

"""
def increase_levels(image):
    minval = np.percentile(image, 2)
    maxval = np.percentile(image, 98)
    image = np.clip(image, minval, maxval)
    image = ((image - minval) / (maxval - minval)) * 255
    return image"""


def pool2d(image, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        image: input 2D array
        kernel_size: int, the FOV of the pooling
        stride: int, the step size of the kernel when sliding through image
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    image = np.pad(image, padding, mode='constant')

    # Window view of image
    new_width = (image.shape[0] - kernel_size) // stride + 1
    new_height = (image.shape[1] - kernel_size) // stride + 1
    output_shape = (new_width, new_height)
    kernel_size = (kernel_size, kernel_size)

    new_shape = output_shape + kernel_size
    stride_x = stride * image.strides[0]
    stride_y = stride * image.strides[1]
    strides_xy = (stride_x, stride_y) + image.strides
    image_formatted = as_strided(image, shape=new_shape, strides=strides_xy)
    image_formatted = image_formatted.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return image_formatted.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return image_formatted.mean(axis=(1, 2)).reshape(output_shape)


# Load configuration file for the entire NEAT process
config_file_name = 'config-feedforward'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file_name)

# Create population based on configurations
p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-248")

# Add reporters to track generational progression
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))  # Save the process after each 10 frames

if __name__ == '__main__':
    pe = neat.ParallelEvaluator(4, eval_genomes) # 4 max
    winner = p.run(pe.evaluate, 1000)
    with open('winner.pk1', 'wb') as output:
        pickle.dump(winner, output, 1)
