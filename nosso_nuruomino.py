# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# ist1109872 Dinis Raimundo da Silva
# ist1106495 Miguel Cordeiro Marques

import sys
from search import Problem, Node
from sys import stdin #In [nuruomino][Board][parse_instance]

class NuruominoState:
    state_id = 0

    def _init_(self, board):
        self.board = board
        self.id = Nuroumino.state_id
        Nuroumino.state_id += 1

    def _lt_(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id

class Board:
    def __init__(self, rows:int, cols:int, grid: list):
        self.rows = rows #pode ser que rows e cols deia jeito
        self.cols = cols
        self.grid = grid
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""

    
    def adjacent_regions(self, region:int) -> list:
        other_regions = []
        sregion = str(region)
        for row_idx, row in enumerate(self.grid):
            for col_idx, cell_value in enumerate(row):
                if cell_value == sregion:
                    adjacents = self.adjacent_positions(row_idx, col_idx)
                    for adj_row, adj_col in adjacents:
                        adj_region = self.grid[adj_row][adj_col]
                        if adj_region[0] < '0' or adj_region[0] > '9':
                            continue #nsei bem, mas se for uma letra L, T, I, S
                        else:
                            adj_region = int(adj_region)
                        if adj_region != region and adj_region not in other_regions:
                            other_regions.append(int(adj_region))

        seen = set()
        return list(filter(lambda x: x not in seen and (seen.add(x) or True), other_regions))
   
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à posição, em todas as direções, incluindo diagonais."""
        
        positions = []

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # skip the original cell
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    positions.append((r, c))
            
        return positions

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à posição, em todas as direções, incluindo diagonais."""

        adj_val = []
        adj_pos = self.adjacent_positions(row, col)
        
        for coord in adj_pos:
            adj_val.append(self.value(coord[0], coord[1]))

        return adj_val
        #TODO
        pass
    
    #others
    
    def value(self, row, col):
        """Devolve o valor da posição"""
        return self.grid[row][col]

    #to create the pieces(4 positions), like after chosing one position the second needs to be adjacent(NO DIAGONALS) to it, etc...
    def adjacent_positions_ND(self, row:int, col:int) -> list:
        """Returns the adjacent positions, NO DIAGONAL"""
    
    #the pieces need to be composed of the region positions
    def get_region(self, region:int) -> list:
        """This function returns the list of positions in the region"""

    #maybe needed we could keep trac of this values for each region in a list for effiency if needed
    def size_region(self, region:int):
        """Returns the number of positions the region has left"""
        return len(self.get_region())

    #dont know very well
    def pieces_region(self, region:int) -> list:
        """Returns the possible pieces of the region"""

    # We need something to check if the given piece creates a square in the board, NOT IF ITS A SQUARE like...
    # I don't know if we make a copy of the board, and place the piece, and then test it with adjacent_values, or we put it on
    # the real board, check it, and afterwards take it away again, putting and taking is probably more efficint both in time and space 
    def check_square(self, piece:list):
        """Checks if the the piece forms a square, if no square returns 1, else returns 0"""

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        print("[nuruomino][Board][parse_instance]: In")
        
        grid = []
        rows = 0 

        line = stdin.readline().split() #ex: ['1', '1', '3', '2', '2', '1']
        cols = len(line) 
        while line != []: #end of file
            rows += 1
            grid.append(line) #vai dar jeito esta em char, pois depois pode estar letras tipo L,T,S,I ou x
            line = stdin.readline().split()
        
        return Board(rows, cols, grid)  
        
    # TODO: outros metodos da classe Board

class Nuruomino(Problem):
    def _init_(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = NuruominoState(board) #acho que é assim
        #TODO
        pass 


    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        #TODO
        pass 

    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        #TODO
        pass 
        

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        #TODO
        pass 

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass


# OTHERS -------------------------------------------------------------------------------------

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


#---------------------------------------------------------------------------------------------

if __name__ == "__main__":

    print(is_what([(0,0),(1,0),(0,1),(1,1)]))

    """
    print("[nuruomino][main]: In")
    board = Board.parse_instance()
    for linha in board.grid:
        print(linha)

    print(board.adjacent_regions(2))
    print(board.adjacent_values(1,1))
    """