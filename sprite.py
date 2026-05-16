import pygame
import pygame.gfxdraw

white = (255, 255, 255)
blue = (0, 0, 255)


class Sprite(pygame.sprite.Sprite):
    def __init__(self, color, radius, initial_position):
        super().__init__()
        self.image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=initial_position)
        self.color = color
        self.radius = radius
        self.speed_x = 2
        self.speed_y = 3
        pygame.gfxdraw.filled_circle(self.image, radius, radius, radius, color)
        pygame.gfxdraw.aacircle(self.image, radius, radius, radius, color)

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        if self.rect.left <= 0 or self.rect.right >= 500:
            self.speed_x *= -1
        if self.rect.top <= 0 or self.rect.bottom >= 500:
            self.speed_y *= -1