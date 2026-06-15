import pygame
from sprite import Sprite
from sprite_demo import Block

pygame.init()

white = (255, 255, 255)
blue = (0, 0, 255)
black = (0, 0, 0)

SCREENWIDTH=1024
SCREENHEIGHT=768
size = (SCREENWIDTH, SCREENHEIGHT)
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
pygame.display.set_caption("Evolution")


# CREATING CANVAS
canvas = pygame.display.set_mode(size)

# TITLE OF CANVAS
pygame.display.set_caption("Evolution")
exit = False

all_sprites_list = pygame.sprite.Group()
# cr = Sprite(blue, 10, (100, 100))
cr = Block(white, 10, (100, 100))
cr.rect.x = 200
cr.rect.y = 200
all_sprites_list.add(cr)

clock=pygame.time.Clock()

while not exit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True

    # Game Logic
    all_sprites_list.update()
    # Drawing on Screen
    # screen.fill(BLACK)
    # Draw The Road
    # pygame.draw.rect(screen, GREY, [40, 0, 200, 300])
    # Draw Line painting on the road
    #pygame.draw.line(screen, WHITE, [140, 0], [140, 300], 5)

    # Now let's draw all the sprites in one go. (For now we only have 1 sprite!)
    all_sprites_list.draw(screen)
    # Refresh Screen
    pygame.display.flip()
    # Number of frames per second e.g. 60

    # screen.fill(black)


    clock.tick(60)

