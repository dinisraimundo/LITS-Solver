# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 94:
# ist1109872 Dinis Raimundo da Silva
# ist1106495 Miguel Cordeiro Santos Miranda Marques

import sys
from search import Problem, Node
from search import depth_first_tree_search;
from search import astar_search;
from sys import stdin #In [nuruomino][Board][parse_instance]

class NuruominoState:
    state_id = 0

    def __init__(self, board, possibilities = [], region = 0, actions = []):
        self.board = board
        self.possibilities = possibilities
        self.region = region
        self.actions = actions
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id

class Board:
    def __init__(self, rows:int, cols:int, grid: list):
        self.rows = rows #pode ser que rows e cols deia jeito
        self.cols = cols
        self.grid = grid
        self.graph = self.build_adjacency_graph()
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""
    def build_adjacency_graph(self):
        number_regions = self.get_number_of_regions()
        graph = [set() for _ in range(number_regions)]

        for row in range(self.rows):
            for col in range(len(self.grid[row])):
                region = int(self.region(row, col)) - 1  # zero-based index
                neighbors = self.adjacent_positions_ND(row, col)

                for nr, nc in neighbors:
                    neighbor_region = int(self.region(nr, nc)) - 1
                    if neighbor_region != region:
                        graph[region].add((nr,nc))

        return graph


    def copy_adjacency_graph(self):
        # Copy list of sets
        return [neighbors.copy() for neighbors in self.graph]


    def print_adjacency_graph(self):
        print("Adjacency Graph:")
        for idx, neighbors in enumerate(self.graph):
            formatted_neighbors = ', '.join(str(n) for n in sorted(neighbors))
            print(f"  Region {idx} -> [{formatted_neighbors}]")

    def print_adjacency_graph_with_coords_and_regions(self):
        print("Adjacency Graph with Coordinates and Regions:")
        for idx, neighbors in enumerate(self.graph):
            formatted_neighbors = []
            for (x, y) in sorted(neighbors):
                r = self.region(x, y)
                formatted_neighbors.append(f"({x},{y}) {r}")
            print(f"  Region {idx+1} -> [{', '.join(formatted_neighbors)}]")



    def update_graph_after_piece(self, piece):
        region = self.region(piece[0][0], piece[0][1])
        number_regions = self.get_number_of_regions()
        
        adjacents = set()
        for row, col in piece:
            for pos in self.adjacent_positions_ND(row, col):
                if region != self.region(pos[0], pos[1]):
                    adjacents.add(tuple(pos))
        
        self.graph[region - 1] = adjacents

        for region_number in range(number_regions):
            if region_number != region:
                to_remove = set()
                for row, col in self.graph[region_number - 1]:
                    if self.region(row, col) == region and (row, col) not in piece:
                        to_remove.add((row, col))
                self.graph[region_number - 1] -= to_remove



    def remove_node_from_adjacency(self, coords, region):
        # node is an int index of the region
        x = coords[0]
        y = coords[1]

        if isinstance(region, str):
            region = (int) (region)

        self.graph[region - 1].remove(coords)
        
        return

    def remove_non_touching_from_region(self, region_cells, piece_cells):
        graph = self.copy_adjacency_graph()

        # Since graph is region-based, region_cells and piece_cells should be lists of region indexes
        # We assume region_cells and piece_cells are region indexes here

        piece_neighbors = set()
        for piece_region in piece_cells:
            # Add neighbors of piece_regions
            piece_neighbors.update(graph[piece_region])

        for region in region_cells:
            if region not in piece_cells and region not in piece_neighbors:
                self.remove_node_from_adjacency(region, graph)

        return graph


    def remove_region_from_adjacency(self, region_cells):
        for region in region_cells:
            self.remove_node_from_adjacency(region, self.graph)


    def are_all_cells_connected(self, cells):
        # cells are region indexes here (adapt to your use case)
        if not cells:
            return False

        graph = self.copy_adjacency_graph()
        visited = set()
        queue = [cells[0]]
        visited.add(cells[0])

        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor in cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(cells)


    def are_all_regions_connected(self):
        visited = set()
        num_regions = self.get_number_of_regions()
        
        
        if num_regions == 0:
            return True

        start = 0
        queue = [start]
        visited.add(start)
        

        while queue:
            current = queue.pop(0)
            for neighbor_tuple in self.graph[current]:
                neighbor = self.region(neighbor_tuple[0], neighbor_tuple[1]) - 1  # Extract neighbor index from tuple

                if neighbor not in visited:
                    visited.add(neighbor)
                    if (len(visited) == 4):
                        return True
                    queue.append(neighbor)

        
        return len(visited) == num_regions
    
    def copy(self):
        new_grid = [row[:] for row in self.grid]
        return Board(self.rows, self.cols, new_grid)
    
    def print(self):
        lits = ["L","I","T","S"]
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                state = self.state(row,col)
                region = self.region(row,col)
                
                if state in lits:
                    line += state
                else:
                    line += str(region)
                if col != self.cols - 1:
                    line += '\t'
            
            if (row == self.rows - 1):
                print(line, end='')
            else:
                print(line)

    def print_raw(self):
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                line += self.grid[row][col]
                if col != self.cols - 1:
                    line += '\t'
            
            if (row == self.rows - 1):
                print(line, end='')
            else:
                print(line)
    
    def adjacent_regions(self, region:int) -> list:
        """Returns the adjacent regions"""
        
        adj_regions = set()
        region_lst = self.get_region(region,True)

        for row,col in region_lst:
            adj_lst = self.adjacent_positions_ND(row, col)
            for adj_row,adj_col in adj_lst:
                adj_r = self.region(adj_row,adj_col)
                adj_s = self.state(adj_row,adj_col)
                if adj_r == region or adj_s == 'X':
                    continue
                adj_regions.add(adj_r)

        return list(adj_regions)
   
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à posição, em todas as direções, incluindo diagonais."""
        
        positions = []

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue  # skip the original cell
                r, c = row + i, col + j
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    positions.append((r, c))
            
        return positions

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à posição, em todas as direções, incluindo diagonais."""

        adj_values = []
        adj_positions = self.adjacent_positions(row, col)
        
        for adj_row,adj_col in adj_positions:
            adj_values.append(self.state(adj_row,adj_col))

        return adj_values

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        
        grid = []
        rows = 0 

        line = stdin.readline().split() #ex: ['1', '1', '3', '2', '2', '1']
        cols = len(line) 
        while line != []: #end of file
            rows += 1
            for i in range(len(line)):
                line[i] = line[i] + ' '
            grid.append(line) #vai dar jeito esta em char, pois depois pode estar letras tipo L,T,S,I ou x
            line = stdin.readline().split()
        
        return Board(rows, cols, grid)  
        
    
    # TODO: outros metodos da classe Board--------------------------------------------------------------------------

    def state(self, row, col):
        """Devolve o valor da posição"""
        return self.grid[row][col][-1]
    
    #its prepared for regions of n > 9, ex: 10 ou 14, etc
    def region(self, row, col):
        """Devolve a regiao da posicao"""
        return int(self.grid[row][col][:-1]) #doesnt include the state(last position of the string)
    
    def set_state(self, row:int, col:int, state:str):
        self.grid[row][col] = self.grid[row][col][:-1] + state

    # the pieces need to be composed of the region positions
    def get_region(self, region:int, all = False) -> list:
        """
        This function returns the list of positions in the region.
        It ignores cells with 'X' or with a complete piece (ex: 'L', 'I', etc...)
        """
        positions = []
        lst = [' ','o']
        if all == True:
            lst.extend(['L','I','T','S'])
        for row in range(self.rows):
            for col in range(self.cols):
                r = self.region(row,col)
                s = self.state(row,col)
                if r == region and s in lst:
                    positions.append((row,col))

        return positions
   
    # to create the pieces(4 positions), like after chosing one position the second needs to be adjacent(NO DIAGONALS) to it, etc...
    def adjacent_positions_ND(self, row: int, col: int) -> list:
        """Returns the adjacent positions (no diagonals): up, down, left, right"""
        adj = []

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                adj.append((r, c))

        return adj


    def adjacent_values_ND(self, row:int, col:int) -> list:
        """Returns the adjacent values to the given position, NO DIAGONAL"""
        
        adj_values = []
        adj_positions = self.adjacent_positions_ND(row,col)

        for adj_row,adj_col in adj_positions:
            adj_values.append(self.state(adj_row,adj_col))

        return adj_values
    
    
    # dont know very well
    def pieces_region(self, region:int) -> list:
        """Returns a list of the possible actions with pieces of the region"""
        
        p1_lst,p2_lst,p3_lst,p4_lst = [],[],[],[]
        pieces = []
        region_lst = self.get_region(region)

        mandatory_lst = []
        for row,col in region_lst:
            if self.state(row,col) == 'o':
                mandatory_lst.append((row,col))
        n = len(mandatory_lst)  
        # n is the number of pieces already selected
        if n == 4:
            p1_lst,p2_lst,p3_lst,p4_lst = [mandatory_lst[0]],[mandatory_lst[1]],[mandatory_lst[2]],[mandatory_lst[3]]
        elif n == 3: 
            p1_lst,p2_lst,p3_lst = [mandatory_lst[0]],[mandatory_lst[1]],[mandatory_lst[2]]
        elif n == 2:
            p1_lst,p2_lst = [mandatory_lst[0]],[mandatory_lst[1]]
        elif n == 1:
            p1_lst = [mandatory_lst[0]]
        elif n == 0:
            p1_lst = region_lst

        """if n == 0:
            p1_lst = region_lst 
        if n > 0:
            p1_lst = [mandatory_lst[0]]
        if n > 1:
            p2_lst = [mandatory_lst[1]]
        if n > 2:
            p3_lst = [mandatory_lst[2]]
        if n == 4:
            p4_lst = [mandatory_lst[3]]"""

        p1_seen = []
        for p1 in p1_lst:
            p1_adj = adjacents_in_ND(p1[0],p1[1],region_lst)
            if n < 2:#only if not defined, do we do atribute lst values
                p2_lst = p1_adj
            
            p1_seen.append(p1)

            p2_seen = p1_seen.copy()
            for p2 in p2_lst:
                if p2 in p1_seen: continue

                p2_adj = adjacents_in_ND(p2[0],p2[1],region_lst)
                if n < 3:#only if not defined, do we do atribute lst values
                    p3_lst = p1_adj + p2_adj

                p2_seen.append(p2)

                p3_seen = p2_seen.copy()
                for p3 in p3_lst:
                    
                    if p3 in p2_seen: continue

                    p3_adj = adjacents_in_ND(p3[0],p3[1], region_lst)
                    if n < 4: #only if not defined, do we do atribute lst values
                        p4_lst = p1_adj + p2_adj + p3_adj
                    
                    p3_seen.append(p3)

                    for p4 in p4_lst:
                        
                        if p4 in p3_seen: continue
                        piece = [p1,p2,p3,p4]
                        form = is_what(piece)
                       
                        
                        #check squares and adjacent pieces
                        if form == 'SQUARE' or self.checks_squares(piece):
                            continue
                        
                        g = board.copy_adjacency_graph()
                        board.update_graph_after_piece(piece)

                        if not self.are_all_regions_connected():
                            board.graph = g
                            continue
                        board.graph = g

                        if self.checks_adjacent_pieces(piece,form):
                            continue
                        #if its a valid piece
                        pieces.append(Action(piece,form,region))
        
        return pieces
    
    def pieces_region2(self, region: int) -> list:
        """Returns the possible tetromino pieces in the region."""
        
        region_list = self.get_region(region)
        pieces = []
        

        for p1 in region_list:
            seen1 = [p1]
            node1 = Node([p1])
            p1_adj = adjacents_in_ND(p1[0], p1[1], region_list)

            for p2 in p1_adj:
                if node1.is_coord_in_ancestors(p2) or p2 in seen1:
                    continue
                seen2 = seen1 + [p2]
                node2 = node1.add_child([p1, p2])
                p2_adj = p1_adj + adjacents_in_ND(p2[0], p2[1], region_list)

                for p3 in p2_adj:
                    if node2.is_coord_in_ancestors(p3) or p3 in seen2:
                        continue
                    seen3 = seen2 + [p3]
                    node3 = node2.add_child(node2.piece + [p3])
                    p3_adj = p2_adj + adjacents_in_ND(p3[0], p3[1], region_list)

                    for p4 in p3_adj:
                        if node3.is_coord_in_ancestors(p4) or p4 in seen3:
                            continue
                        piece4 = node3.piece + [p4]
                        if not is_square(piece4) or not self.checks_squares(piece4):
                            pieces.append(piece4)

        return pieces  # remove duplicates

    def is_connected(self, cells):
        """Check if all the cells in the piece are connected (using BFS)"""
        if not cells:
            return False

        seen = set()
        queue = [cells[0]]
        seen.add(cells[0])

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))

        return len(seen) == len(cells)

    def generate_combinations(self, lst, k):
        """Manual replacement for itertools.combinations"""
        def backtrack(start, path):
            if len(path) == k:
                result.append(path[:])
                return
            for i in range(start, len(lst)):
                path.append(lst[i])
                backtrack(i + 1, path)
                path.pop()
        result = []
        backtrack(0, [])
        return result

    def pieces_region3(self, region: int) -> list:
        """Return all valid 4-cell connected pieces in a region"""
        region_cells = self.get_region(region)
        mandatory = [(r, c) for r, c in region_cells if self.state(r, c) == 'o']
        n = len(mandatory)

        if n > 4:
            return []

        if n == 4:
            candidates = [mandatory]
        else:
            remaining = [cell for cell in region_cells if cell not in mandatory]
            needed = 4 - n
            candidates = [
                mandatory + combo
                for combo in self.generate_combinations(remaining, needed)
            ]

        valid_actions = []
        for piece in candidates:
            if not self.is_connected(piece):
                continue
            form = is_what(piece)
            if form == 'SQUARE' or self.checks_squares(piece):
                continue
            if self.checks_adjacent_pieces(piece, form):
                continue
            valid_actions.append(Action(piece, form, region))

        return valid_actions      
              

    # We need something to check if the given piece creates a square in the board, NOT IF ITS A SQUARE like...
    # I don't know if we make a copy of the board, and place the piece, and then test it with adjacent_values, or we put it on
    # the real board, check it, and afterwards take it away again, putting and taking is probably more efficint both in time and space 
    def checks_squares(self, piece:list):
        """Checks if putting the piece in the board forms a square, if square returns 1, else returns 0"""
        
        original = []
        
        isSquare = False

        #free states
        free = [' ', 'X'] 

        #this puts the form in the board as 'o' temporarly, so we save the old values
        for row,col in piece:
            original.append(self.state(row,col))
            self.set_state(row,col,'o')
        
        for row,col in piece:
            adj_val = self.adjacent_values(row,col)
            # Acontece quando é um quadrado
            # Para quando não está na fronteira
            
            if (len(adj_val) == 3):
                if not ((adj_val[0] in free) or (adj_val[1] in free) or (adj_val[2] in free)):
                    isSquare = True
                

            elif (len(adj_val) == 5): 
                if (col == 0 or col == self.cols-1) and \
                    (not((adj_val[0] in free) or (adj_val[1] in free) or (adj_val[2] in free)) or \
                     not((adj_val[2] in free) or (adj_val[3] in free) or (adj_val[4] in free))):
                        isSquare = True
                
                elif (row == 0) and \
                    (not((adj_val[0] in free) or (adj_val[2] in free) or (adj_val[3] in free)) or \
                     not((adj_val[1] in free) or (adj_val[3] in free) or (adj_val[4] in free))):
                        isSquare = True

                elif (row == self.rows-1) and \
                    (not ((adj_val[0] in free) or (adj_val[1] in free) or (adj_val[3] in free)) or \
                     not ((adj_val[1] in free) or (adj_val[2] in free) or (adj_val[4] in free))):
                        isSquare = True
                

            elif len(adj_val) == 8 and \
                ((not ((adj_val[0] in free) or (adj_val[1] in free) or (adj_val[3] in free))) \
                or (not((adj_val[1] in free) or (adj_val[2] in free) or (adj_val[4] in free))) \
                or (not((adj_val[3] in free) or (adj_val[5] in free) or (adj_val[6] in free))) \
                or (not((adj_val[4] in free) or (adj_val[6] in free) or (adj_val[7] in free)))):
                    isSquare = True
        
        counter = 0
        for row,col in piece:
            self.set_state(row,col,original[counter])
            counter += 1

        return isSquare
    

    def checks_adjacent_pieces(self, piece:list, form = ''):
        """Checks if there are adjacent pieces iqual to the given piece. If there are adjacent pieces return 1, else 0"""
        
        if form == '':
            form = is_what(piece)

        for row,col in piece:
            for adj_row, adj_col in self.adjacent_positions_ND(row,col):
                if self.state(adj_row,adj_col) == form and (adj_row, adj_col) not in piece:
                    return True
        return False

    # Devolve as coordenadas de interseção das peças dadas
    def no_pieces(self, pieces:list, zone:list)->list:
        """Returns the list of coordenates in the zone which have no pieces"""
        x_list = zone.copy()

        for piece in pieces:
            for coord in piece:
                if coord in x_list:
                    x_list.remove(coord)
        return x_list

    def piece_intersection(self, pieces:list)->list:
        """Returns a list of the positions in the region that all pieces use."""
        # pieces -> list of lists
        if len(pieces) == 0:
            return []
        intr = set(pieces[0])
        for piece in pieces[1:]:
            intr = intr & set(piece)

        # nunca tinha pensado nesta de iterar de trás para a frente
        """intr = pieces[0] # Usar peça random funciona porque interseção é sub-conjunto de qualquer peça

        for i in range(1, len(pieces)):
            for j in range(len(intr) - 1, -1, -1): # Iterar de tras para frente para nao os pops nao afetarem
                if intr[j] not in pieces[i]:
                    intr.pop(j)
        """
        return intr
        
    
    def get_number_of_regions(self):
        regioes = set()
        
        for row in range(self.rows):
            for col in range(self.cols):
                regioes.add(self.region(row,col))
        return len(regioes)

    def get_number_of_incompleted_regions(self):
        regioes = set()
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.state(row,col) in [' ', 'o']:
                    regioes.add(self.region(row,col))
        return len(regioes)

    def set_positions_to_state(self, positions:list, state:str):
        for row,col in positions:
            self.set_state(row,col,state)

    #try not to call this function, highly unefficient
    def set_all_mandatory_pieces(self):
        number_of_regions = self.get_number_of_regions()
        for i in range(number_of_regions):
            self.set_positions_to_state(self.piece_intersection(self.pieces_region(i + 1)), 'o')
    
    # Retorna a peca numa regiao
    def piece_in_region(self, region, accept_o = False, p = False):
        """
        This function returns a piece in the region
        If accept_o = True it consideres a piece made of 4 o's a piece (needed for check_board_impossible)
        If p == True enables  specific prints in the function (used in check_board_impossible)
        """
        coords_count = 0 # para acabar mais cedo
        piece = []
        reject = [' ', 'X']
        if not accept_o: #if o isn't considered a piece, add to the regect list
            reject.append('o')
        for i in range(self.rows):
            for j in range(self.cols):
                if p and self.region(i,j) == 11:
                    print(self.state(i,j))
                if self.region(i,j) == region and self.state(i,j) not in reject:
                    coords_count = coords_count + 1
                    piece.append((i,j))
                if coords_count == 4:
                    break
            if coords_count == 4:
                break
        return piece
    
    # UNDER CONSTRUCTION -------
    
    #mad sus. Probably not completed xD Prob not completed fr
    def set_all_impossible_pieces(self):
    
        number_of_regions = self.get_number_of_regions()

        # Iterar pelo tabuleiro todo e verificar nas coordenadas disponíveis ao colocar lá algo forma um quadrado
        for row in range(self.rows):
            for col in range(self.cols):
                if (self.state(row,col) != 'o'): # Não está nas peças obrigatórias
                    pass
    

    def set_impossible_pieces(self, positions:list):
        pass

    # UNDER CONSTRUCTION ------- END
    def forms_square(self, row, col, putting = False):
        """ 
        By default function checks if (row,col) belongs to a square, it returns true if so and false otherwise.
        If putting = True, the function fills (row,col) on the board with 'o' and then checks if now it forms a square,
        returning true if forms a square and false otherwise.
        """
        rows = len(self.grid)
        cols = len(self.grid[0])
        free = [' ', 'X']

        # Possíveis quadrados que incluem (i, j)
        offsets = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, -1), (0, 0), (1, -1), (1, 0)],
            [(-1, 0), (-1, 1), (0, 0), (0, 1)],
            [(-1, -1), (-1, 0), (0, -1), (0, 0)]
        ]

        state = ''
        if putting:
            state = self.state(row,col)
            self.set_state(row,col, 'o')

        # Mudar a logic aaqui
        for square in offsets:
            positions = [(row + dr, col + dc) for dr, dc in square]

            if all(0 <= r < rows and 0 <= c < cols for r, c in positions):  
                if all(self.state(r,c) not in free for r, c in positions):
                    if putting:
                        self.set_state(row,col, state)
                    return True

        if putting:
            self.set_state(row,col, state)

        return False

    def colateral_Xs(self, piece:list, region:int = -1)->list:
        """
        This function returns the positions adajcent to the piece, outside of the piece region that
        need X because it's now a potencial square
        """
        if region == -1:
            region = self.region(piece[0][0],piece[0][1])

        adj_exterior = set()
        for row,col in piece:
            for adj_row, adj_col in self.adjacent_positions(row,col):
                if self.region(adj_row,adj_col) != region:
                    adj_exterior.add((adj_row,adj_col))
        
        adj_X = []

        for row,col in adj_exterior:
            if (self.forms_square(row,col,True)):
                adj_X.append((row,col))
        return adj_X
    
    def check_board_impossible(self, possibilities:list, n_regions:int):
        """This function returns True if the board is impossible to solve."""
        for r in range(n_regions):
            if possibilities[r] == 0 and self.piece_in_region(r+1,accept_o=True) == []:
                #print("here -> " + str(r+1) + " " + str(possibilities) + " ")
                return True
        return False

    def are_regions_hungry(self,adjacentes,n_grupos):
        """Returns True if theres no way (no combo) for the grups the coexist with the possible modules. Else Returns False"""
        for combo in self.combos(adjacentes):
            if len(combo) == n_grupos:
                return False

    def combos(self, modules:list):
             
        if modules == []:
            return [[]]
        combos_lst = []
    
        line = modules[0]
        combos = self.combos(modules[1:])

        for module in line:
            for combo in combos:
                combos_lst.append(module + combo)

        return combos_lst
    
    def region_of_piece(self,piece:list):
        return self.region(piece[0][0],piece[0][1])

    def adjacent_regions_to_piece(self, piece:list):
        region = self.region_of_piece(piece)
        #print("region = " + str(region))
        adj_regions = set()
        for row,col in piece:
            for adj_row,adj_col in self.adjacent_positions_ND(row,col):
                adj_region = self.region(adj_row,adj_col)
                adj_state = self.state(adj_row,adj_col)
                if adj_region != region and adj_state != 'X':
                    adj_regions.add(adj_region)
        #print("adj = " + str(adj_regions))
        return adj_regions
    
    def get_modules(self,possibilities:list, actions:list, n_regions:int, groups:list):
        modules_lst = []
        #for each uncompleted region(ponto de conexao)
        for region in range(1,n_regions+1):
            if possibilities[region-1] == 0:
                continue
            
            #lista de regioes adjacentes ao ponto de recolha
            adj_lst = []
            for adj in self.adjacent_regions(region):
                if possibilities[adj-1] == 0:
                    adj_lst.append(adj)
            if adj_lst == []:
                continue
            #print("\n")
            #print("possivel ponto de conexao = " + str(region))
            #print("possivel adj do ponto de conexao = " + str(adj_lst))
            #lista de modulos que o ponto de recolha podera escolher para conectar o maximo de grupos
            modules = []
            #cada modulo corresmonde a uma peça no tabuleiro
            for action in actions[region-1]:
                piece = action.positions
                adj_regions = self.adjacent_regions_to_piece(piece)
                if adj_regions == []:
                    continue
                
                module = []
                for group in groups:
                    if (group & adj_regions): #se alguma das regioes do grupo for adjacente ao modulo, adicionar o grupo ao modulo
                        module.append(group)
                        #print("ponto de conexao = " + str(region) + " salva " + str(group))
                if module != [] and (module not in modules):
                    modules.append(module)
            
            if modules != []:
                modules_lst.append(modules)

        return modules_lst
    
    def get_complete_groups(self,possibilities:list, n_regions:int):
        groups = []
        seen = set()
        for region in range(1,n_regions+1):
            if possibilities[region-1] != 0 or region in seen:
                continue

            group = set()
            queue = [region]
            seen.add(region)

            while queue:
                current = queue.pop(0)
                group.add(current)
                piece = self.piece_in_region(current)
                for adj in self.adjacent_regions(current):
                    if possibilities[adj-1] == 0 and adj not in seen:
                        adj_piece = self.piece_in_region(adj)
                        if adjacents_ND(adj_piece, piece):
                            queue.append(adj)
                            seen.add(adj)
            
            groups.append(group)
        return groups

    def check_impossible_connected_board2(self,possibilities:list, actions:list, n_regions:int):
        groups = self.get_complete_groups(possibilities, n_regions)
        #print("groups = ", groups)
        if groups == []:
            return False
        modules = self.get_modules(possibilities, actions, n_regions, groups)
        #print("MODULEs")
        #for line in modules:
        #    print(line)
        combos = self.combos(modules)
        #print("COMBOS")
        #for combo in combos:
            #print(combo)

        for combo in combos:
            #if theres a combo with all groups in it, (satisfied)
            count = 0
            for group in groups:
                #print(group)
                #print(combo)
                #print("in combo = " + str(group in combo))
                if group in combo:
                    count += 1
            if count == len(groups):
                return False
        
        return True
    
    def check_impossible_connected_board(self,possibilities:list, actions:list, n_regions:int):
        done_regions = []
        for r in range(1,n_regions+1):
            if possibilities[r-1] == 0:
                done_regions.append(r)        
        checked = []
        for r in done_regions:
            if r in checked:
                continue
            
            
            adj_regions = set()
            checked.append(r)

            regions = [r]
            positions  = [self.piece_in_region(r)] #no sure if accept_o = True would be better
            i = 0
            while True:
                if i >= len(regions):
                    break
                region = regions[i]
                piece = positions[i]
                adj_regions1 = self.adjacent_regions(region)
                for adj_r in adj_regions1:

                    if adj_r not in done_regions:
                        adj_regions.add(adj_r)
                        continue
                    elif adj_r in regions:
                        continue

                    adj_piece = self.piece_in_region(adj_r)
                    if adjacents_ND(adj_piece, piece):
                        checked.append(adj_r)
                        regions.append(adj_r)
                        positions.append(adj_piece)
                i += 1

            group_positions = []
            for piece in positions:
                group_positions +=  piece
            
            actions_positions= []
            for adj_r in adj_regions:
                for action in actions[adj_r-1]:
                    actions_positions+= action.positions
                
            #se algum grupo isolado não tem adjacentes com açoes return False
            if adjacents_ND(group_positions,actions_positions) == False:
                return True
            
        return False
    
    def remove_impossible_acions(self, actions, n_regions):
        #SECOND CHECK, UPTADE ACTIONS
        for r in range(n_regions):
            adj_regions = self.adjacent_regions(r + 1)
            region_actions = actions[r]
            if region_actions == []:
                continue
            removed_actions = []
            for action in region_actions:
                
                piece = action.positions
                form = action.state
                original = []
                for row,col in piece:
                    original.append(self.state(row,col))
                    self.set_state(row,col,form)

                
                for adj_r in adj_regions:
                    
                    adj_actions = actions[adj_r-1]
                    if adj_actions == []: #completed regions
                        continue
                    
                    rejected = True

                    for adj_action in adj_actions:
                        adj_piece = adj_action.positions
                        adj_form = adj_action.state
                        if not adjacents_ND(piece, adj_piece):
                            rejected = False
                            break #this region is compatible with piece, pass to the next region
                        join = piece + adj_piece

                        if self.checks_squares(join) == False \
                            and self.checks_adjacent_pieces(adj_piece, adj_form) == False: #if no squares
                            rejected = False
                            break

                    if rejected == True: #rejected by at least a region
                        removed_actions.append(action)
                        break
                
                for row,col in piece:
                    self.set_state(row,col,original.pop(0)) #don't think there's any problem

            for action in removed_actions:
                region_actions.remove(action)
        #END SECOND CHECK


    def all_pieces_connected(self):
        lits = ['L', 'I', 'T', 'S']
        regions = set()
        n_regions = board.get_number_of_regions()
        seen = []
        frontier = self.piece_in_region(1)
        if frontier == []:
            return False
        
        regions.add(1)

        while frontier != []:
            row,col = frontier.pop(0)
            seen.append((row,col))
            for adj_row,adj_col in self.adjacent_positions_ND(row,col):
                state = self.state(adj_row,adj_col)
                region = self.region(adj_row,adj_col)
                
                if state in lits: #if letter
                    regions.add(region) #add region to connected
                    #if new add to frontier
                    if (adj_row,adj_col) not in seen and (adj_row,adj_col) not in frontier:
                        frontier.append((adj_row,adj_col))
        return n_regions == len(regions)

class Action:
    def __init__(self, positions:list, state:str, region:int):
        self.positions = positions
        self.state = state
        self.region = region
        pass
    
    def print(self):
        print([self.positions],end=" ")


class Nuruomino(Problem):
    #Eu é que coloquei isto e isto é mad sus, mas na inicialização
    #  de NuruominoState isto ta a ser chamado, por isso coloquei aqui

    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = NuruominoState(board) #acho que é assim
        #TODO
        pass 
    
    # ignora
    def actions_region(self, board:Board, region:int):
        """This function returns a region actions"""
        pass # passei o código para a actions, tava a dar mais jeito
        
    
    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        #TODO  
        
        
        
        board:Board= state.board #just so vscode knows that its a Board and help with sugestions in the methods
        board_copy = board.copy()
        n_regions = board.get_number_of_regions()
        
        actions = []
        possibilities = []
        if state.actions == []:
            actions = [[] for _ in range(n_regions)] 
            possibilities = [0 for _ in range(n_regions)]
        else:
            actions = state.actions
            possibilities = state.possibilities

        #PRINTS
        """print()
        print("################################################")
        print("id = " + str(state.id) + ">")
        print("BOARD: " + str(state.id) + " Region: " + str(state.region))
        board.print_raw()
        print(possibilities)
        print("OLD ACTIONS:")
        print_actions(actions)"""
        

        if state.actions == []:
            for r in range(1, n_regions+1):
                region_actions = board.pieces_region(r)
                actions[r-1] = region_actions
                possibilities[r - 1] = len(region_actions)
        else:
            for r_idx in range(n_regions):
                #print(len(actions[r_idx]))
                remove = []
                for action in actions[r_idx]:
                    #print(len(actions[r_idx]))
                    if state.id == 8:
                        if r_idx == 8:
                            
                            val  =board.checks_squares(action.positions)
                            #print("ativa ? " + str(val))
                            #print("peca = " + str(action.positions))
                            #print("form = " + str(action.state))
                    if board.checks_squares(action.positions) or\
                        board.checks_adjacent_pieces(action.positions,action.state):
                        #print(action.positions)
                        remove.append(action)
                    else:
                        for row,col in action.positions:
                            if board.state(row,col) not in [' ','o']:
                                remove.append(action)
                                break
                    #print("end = " + str(len(actions[r_idx])))
                for action in remove:
                    actions[r_idx].remove(action)
                possibilities[r_idx] = len(actions[r_idx])

        

        intr_positions = []
        no_pieces = []
        for r in range(1, n_regions+1):
             # Pecas possiveis nessa regiao

            regiao_lst = board.get_region(r) # Lista de coordenadas de uma regiao
            region_actions = actions[r-1]

            pieces = [] # needed for the next lines
            for action in region_actions:
                pieces.append(action.positions)

            intr_positions = board.piece_intersection(pieces) # Coloca os o
            no_pieces = board.no_pieces(pieces, regiao_lst) # Coloca os X
        
            board.set_positions_to_state(intr_positions, 'o')
            #convert external potencial squares to x, for example in t piece, theres two positions who are potencial squares
            board.set_positions_to_state(board.colateral_Xs(intr_positions, r),'X')
            board.set_positions_to_state(no_pieces, 'X')
            remove = []
            for action in region_actions:
                for row,col in action.positions:
                    if board.state(row,col) not in [' ','o']:
                        remove.append(action)
                        break
            for action in remove:
                region_actions.remove(action)

        
        
        state.possibilities = possibilities
        print(possibilities)
        #from time import sleep
        #sleep(10000)
        state.actions = actions

        """print("##############")
        print("id = " + str(state.id) + ">")
        print("BOARD: " + str(state.id) + " Region: " + str(state.region))
        board.print_raw()
        print(possibilities)
        print("NEW ACTIONS:")   
        print_actions(actions)"""
        
        
        """#FILTER ACTIONS by another method
        board.remove_impossible_acions(actions, n_regions)
        for i in range(len(actions)):
            possibilities[i] = len(actions[i])"""

        idx = idx_of_lowest(possibilities)
        
        #print("X1")
        #if last action caused a impossible board, end its actions
        #if (board.check_board_impossible(possibilities,n_regions)):
            #print("board_impossible")
        #    return []
        #print("X2")
        actions_copy = [[action for action in region_actions] for region_actions in state.actions]
        possibilities_copy = possibilities.copy()
        if (board.check_impossible_connected_board2(possibilities_copy, actions_copy, n_regions)):
            #print("board_impossible_connected")
            return []
        #print("X3")
        
        

        #print("END ACTIONS\n\n\n")
        return actions[idx]
        

    def result(self, state: NuruominoState, action: Action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        
        
        copy_board = state.board.copy()
        
        piece = action.positions
        form = action.state

        region = copy_board.region(piece[0][0],piece[0][1])

        actions_copy = [[action for action in region_actions] for region_actions in state.actions]
        actions_copy[region-1] = [] #no more duture actions for the action region
        copy_possibilities = state.possibilities.copy()
        copy_possibilities[region-1] = 0

        copy_board.set_positions_to_state(piece, form)
        copy_board.set_positions_to_state(copy_board.get_region(region), 'X')
        copy_board.set_positions_to_state(copy_board.colateral_Xs(piece, region), 'X')
        #print("------------------------------------------------>")
        
        state = NuruominoState(copy_board,copy_possibilities, region, actions_copy)
        state.board.graph = state.board.update_graph_after_piece(piece)
        
        #print("id = " + str(state.id) + ">")
        #print("region -> " + str(region))
        #copy_board.print_raw()
        #print(state.possibilities)
        #print("------------------------------------------------<")

        return state

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        board:Board = state.board
        n_regions = board.get_number_of_regions()
        

        # Alguns destes constraints podem estar a ser automaticamente executados ao colocar as pecas mas vou colocar aqui e depois podemos
        # retirar estes

        if (not board.are_all_regions_connected):
            return False

        if(board.all_pieces_connected() == False):
            return False
        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        state = node.state


        if state.id == 0:
            return 0

        action= node.action
        
        #piece = action.positons
        #form = action.state
        
        possibilities = state.possibilities
        region = action.region

        path_cost = node.path_cost
        n_incomplete = state.board.get_number_of_incompleted_regions()
        
        return n_incomplete * 10000 + 10 * possibilities[region-1]
        return (len(possibilities) - state.board.get_number_of_incompleted_regions()) + (100 * (possibilities[region - 1] - 1))

        
        pass


# OTHERS -------------------------------------------------------------------------------------
class Node:
    def __init__(self, piece, parent=None):
        self.piece = piece            # list of coords
        self.parent = parent
        self.children = []

    def add_child(self, child_piece):
        child = Node(child_piece, parent=self)
        self.children.append(child)
        return child
    
    def __repr__(self):
        return f"Node({self.value})"
    
    def is_coord_in_ancestors(node, coord):
        current = node
        while current:
            if coord in current.piece:
                return True
            current = current.parent
        return False

def print_list_of_actions(actions_lst:list):
    for action in actions_lst:
        action.print()
        
def print_actions(actions:list):
    for r in range(1,len(actions) +1):
        print(str(r) + " -> ", end="")
        print_list_of_actions(actions[r-1])
        print()

def adjacents_ND(pos1:list, pos2:list):
    """
    This function sees if the two groups of positions are adjacent or not.
    Returning True if adjacente and False otherwise
    """
    for row,col in pos1:
        if adjacents_in_ND(row,col,pos2) != []:
            return True
    return False


#It can be helpfull to several things, and it will be usefull in these next functions of is_
def adjacents_in(row:int, col:int, positions:list):
    """Returns the adjacent positions of the given position <pos> in the given positions"""
    adj = []
    adj.extend(adjacents_in_ND(row,col,positions))
    for i in [-1,1]:
        if (row + i,col + i) in positions:
            adj.append((row+i,col))
        if (row + i, col - i) in positions:
            adj.append((row, col+i))
    
    return adj

#NO DIAGONALS
def adjacents_in_ND(row:int, col:int, positions:list):
    """Returns the adjacent positions of the given position <pos> in the given positions"""
    adj = []

    for i in [-1,1]:
        if (row + i,col) in positions:
            adj.append((row+i,col))
        if (row, col + i) in positions:
            adj.append((row, col+i))
    
    return adj

def is_what(piece:list):
    #returns 'L' OR 'I' OR 'T' OR 'S' OR 'SQUARE'
    
    count = [0,0,0]
    count_ND = [0,0,0] #adj(NAO DIAGONAL)
    
    for row,col in piece:
        count[len(adjacents_in(row,col,piece)) - 1] += 1

    for row,col in piece:
        count_ND[len(adjacents_in_ND(row,col,piece)) - 1] += 1

    if count[2] == 4:
        return 'SQUARE'
    elif count[0] == 1:
        return 'L'
    elif count[0] == 2:
        return 'I'
    elif count[2] == 2:
        if count_ND[0] == 3:
            return 'T'
        elif count_ND[0] == 2:
            return 'S'
    
    return 1 #error

#estes metodos de dar check atravez de adjacente são bue sus mas funciona, testei no papel.
# Basicamente dá para saber que peça é pelo numero de adjacentes que as peças tem entre si
def is_square(piece:list):        
    return is_what(piece) == 'SQUARE'

def is_L(piece):
    return is_what(piece) == 'L'

def is_I(piece):
    return is_what(piece) == 'I'

def is_T(piece):
    return is_what(piece) == 'T'

def is_S(piece):
    return is_what(piece) == 'S'

def idx_of_lowest(lst:list):
    """This function returns the index of the lowest element grather  than zero.
    There cant be no negative elements"""
    idx = 0
    min = 0
    if max(lst) != 0:
        for i in range(len(lst)):
            val = lst[i]
            
            if val == 1:
                min = 1
                idx = i
                break

            elif val > 1:
                if min == 0:
                    min = val
                    idx = i
                elif val < min:
                    min = val
                    idx = i
    return idx

#---------------------------------------------------------------------------------------------

if __name__ == "__main__":

    board = Board.parse_instance()
    problem = Nuruomino(board)

    solution = depth_first_tree_search(problem)  # Run DFS
    if solution:
        solution.state.board.print()
