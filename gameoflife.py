import pygame
import random

pygame.init()

BLACK = (0, 0, 0)
GREY = (128, 128, 128)
YELLOW = (255, 255, 0)

WIDTH, HEIGHT = 700, 700
TILE_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = WIDTH // TILE_SIZE, HEIGHT // TILE_SIZE
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()

def gen(num):
    return set([(random.randrange(0, GRID_HEIGHT), random.randrange(0, GRID_WIDTH)) for _ in range(num)]) # convert to set to remove duplicates in case of random.randrange(0, GRID_HEIGHT) == random.randrange(0, GRID_WIDTH)

def draw_grid(positions): # positions is a set of tuples
    for position in positions: # for each living cell (x, y)
        col, row = position  # (x, y)
        top_left = (col * TILE_SIZE, row * TILE_SIZE) # (x, y)
        pygame.draw.rect(screen, YELLOW, (*top_left, TILE_SIZE, TILE_SIZE)) # draw rectangle at (x, y) with width and height of TILE_SIZE

    for row in range(GRID_HEIGHT):
        pygame.draw.line(screen, BLACK, (0, row * TILE_SIZE), (WIDTH, row * TILE_SIZE))
    for col in range(GRID_WIDTH):
        pygame.draw.line(screen, BLACK, (col * TILE_SIZE, 0), (col * TILE_SIZE, HEIGHT))

def adjust_grid(positions):
    all_neighbours = set() # store all neighbours of all living cells
    new_positions = set() # store new living cells

    for position in positions: # collect all neighbours of living cells
        neighbours = get_neighbours(position) # get all 8 possible neighbours
        all_neighbours.update(neighbours) # add neighbours to set

        neighbours = list(filter(lambda x: x in positions, neighbours)) # filter out dead neighbours, filter gives iterator, list() to convert to list

        if len(neighbours) in [2, 3]: # if 2 or 3 neighbours, cell lives
            new_positions.add(position) # add to new living cells

    for position in all_neighbours: # loop through neighbours of all living cells
        neighbours = get_neighbours(position) # get all 8 possible neighbours
        neighbours = list(filter(lambda x: x in positions, neighbours)) 

        if len(neighbours) == 3: # if 3 neighbours, cell is born
            new_positions.add(position)

    return new_positions

def get_neighbours(pos):
    x, y = pos
    neighbours = []
    for dx in [-1, 0, 1]:
        if x + dx < 0 or x + dx >= GRID_WIDTH: # if out of bounds, not a neighbour
            continue
        for dy in [-1, 0, 1]:
            if y + dy < 0 or y + dy >= GRID_HEIGHT: # if out of bounds, not a neighbour
                continue
            if dx == 0 and dy == 0: # if same cell,
                continue
            neighbours.append((x + dx, y + dy)) # add to neighbours

    return neighbours

def main():
    running = True
    playing = False
    count = 0
    update_freq = 30
    
    positions = set() # store living cells

    while running:
        clock.tick(FPS)

        if playing:
            count += 1

        if count >= update_freq:
            count = 0
            positions = adjust_grid(positions)

        pygame.display.set_caption(f'Game of Life - FPS: {int(clock.get_fps())} - Playing: {playing} - Update Frequency: {update_freq} - Living Cells: {len(positions)}')
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col, row = x // TILE_SIZE, y // TILE_SIZE
                pos = (col, row)
                if pos in positions:
                    positions.remove(pos)
                else:
                    positions.add(pos)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: # Space for pause
                    playing = not playing
                
                if event.key == pygame.K_c: # C for clear
                    positions = set()
                    playing = False

                if event.key == pygame.K_g:
                    positions = gen(random.randrange(2,5) * GRID_WIDTH) # G for generate

        screen.fill(GREY)
        draw_grid(positions)
        pygame.display.update()


    pygame.quit()

if __name__ == '__main__':
    main()