import pygame
import random
import numpy as np


# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class Block(pygame.sprite.Sprite):
    """
    This class represents the ball.
    It derives from the "Sprite" class in Pygame.
    """

    def __init__(self, color, width, height):
        """ Constructor. Pass in the color of the block,
        and its x and y position. """

        # Call the parent class (Sprite) constructor
        super().__init__()
        self.speed_x = 1
        self.speed_y = 1
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image_original = pygame.image.load("img/boby_blue.png").convert()
        self.image = self.image_original
        self.image_rotate()
        self.image.set_colorkey(WHITE)


        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values
        # of rect.x and rect.y
        self.rect = self.image.get_rect()

    def image_rotate(self):
        # Calculate rotation angel from current speed
        rad = np.arctan2(self.speed_y, self.speed_x)
        degrees = int(rad * 180 / np.pi)
        if degrees < 0:
            degrees = 360 + degrees
        self.image = pygame.transform.rotate(self.image, degrees)


    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        if self.rect.left <= 0 or self.rect.right >= 500:
            self.speed_x *= -1
            self.image_rotate()
        if self.rect.top <= 0 or self.rect.bottom >= 500:
            self.speed_y *= -1
            self.image_rotate()