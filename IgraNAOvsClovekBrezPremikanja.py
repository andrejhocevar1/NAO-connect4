#import random
import sys
import time
import os
import math
from optparse import OptionParser

from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule

import cv2
import numpy as np

#KONSTANTE
NAO_IP = "192.168.0.130"
Spremenljivka = None
KONEC = False
N_STEPS = 7

"""
def random(board):
     #tabela, ki ustreza formatu predict
    b=[]
    for j in range(6):
        b.append([])
        for k in range(7):
            b[j].append([])
            b[j][k].append([])   
            b[j][k][0]=board[j][k]   
    # Use the best model to select a column
    col, _ = model.predict(b)
    # Check if selected column is valid
    is_valid = (board[5][int(col)] == 0)
    # If not valid, select random move. 
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(7) if board[5][int(col)] == 0])
    #zacasno deluje po principu random
    return random.choice([col for col in range(7) if board[0][int(col)] == 0])
"""

#Nastavitve parametrov za zaznavanje krogov
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 250
params.thresholdStep = 5
params.filterByArea = True
params.minArea = 600
params.filterByCircularity = True
params.minCircularity=0.7
params.filterByInertia = True
params.minInertiaRatio=0.4
params.filterByConvexity = True
params.minConvexity = 0.8

#konstante igralcev
#PRAZNO=0
RDECA=1
RUMENA=2

#FUNKCIJE
#fotografiraj
def fot(foto):
    foto.takePicture("/home/nao/recordings/cameras","image.jpg")
    cmd = 'pscp -pw 2UglyBetty nao@192.168.0.130:/home/nao/recordings/cameras/image.jpg .'
    os.system(cmd)
    return cv2.imread("image.jpg")

#funkcija za urejanje
def fun(e): return e[1]

#ali smo zmagali
def winning_move(board, piece): 
    # Check horizontal locations for win
     for c in range(4):
        for r in range(6):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
 
    # Check vertical locations for win
     for c in range(7):
        for r in range(3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
 
    # Check positively sloped diaganols
     for c in range(4):
        for r in range(3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
 
    # Check negatively sloped diaganols
     for c in range(4):
        for r in range(3, 6):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
     return False

#ali obstaja mozna poteza
def can_move(board):
    for x in range(7):
            if board[0][x]==0:
                return True
    return False

#FUNKCIJE ZA DOLOCANJE POTEZE
#logika za potezami
def move(grid, mark):
    # Get list of valid moves
    valid_moves = [c for c in range(7) if grid[0][c] == 0]
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, mark, 3) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return max_cols[int(len(max_cols)/2)]

# Uses minimax to calculate value of dropping piece in selected column
def score_move(grid, col, mark, nsteps):
    next_grid = drop_piece(grid, col, mark)
    score = minimax(next_grid, nsteps-1, False, mark)
    return score

# Helper function for minimax: checks if agent or opponent has four in a row in the window
def is_terminal_window(window):
    return window.count(1) == 4 or window.count(2) == 4

# Helper function for minimax: checks if game has ended
def is_terminal_node(grid):
    # Check for draw 
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(6):
        for col in range(4):
            window = list(grid[row, col:col+4])
            if is_terminal_window(window):
                return True
    # vertical
    for row in range(3):
        for col in range(7):
            window = list(grid[row:row+4, col])
            if is_terminal_window(window):
                return True
    # positive diagonal
    for row in range(3):
        for col in range(4):
            window = list(grid[range(row, row+4), range(col, col+4)])
            if is_terminal_window(window):
                return True
    # negative diagonal
    for row in range(3, 6):
        for col in range(4):
            window = list(grid[range(row, row-4, -1), range(col, col+4)])
            if is_terminal_window(window):
                return True
    return False

# Minimax implementation
def minimax(node, depth, maximizingPlayer, mark):
    is_terminal = is_terminal_node(node)
    valid_moves = [c for c in range(7) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return get_heuristic(node, mark)
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark)
            value = max(value, minimax(child, depth-1, False, mark))
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1)
            value = min(value, minimax(child, depth-1, True, mark))
        return value

#funkcija, ki izvede potezo
def drop_piece(grid, col, mark):
    next_grid = np.copy(grid)
    for row in range(5, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid

#hevristicen nacin racunanja vrednosti poteze
def get_heuristic(grid, mark):
    num_threes = count_windows(grid, 3, mark)
    num_fours = count_windows(grid, 4, mark)
    num_threes_opp = count_windows(grid, 3, mark%2+1)
    num_fours_opp = count_windows(grid, 4, mark%2+1)
    score = num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
    return score

#doloci, ce okno ustreza hevristicnim pogojem
def check_window(window, num_discs, piece):
    return (window.count(piece) == num_discs and window.count(0) == 4-num_discs)
    
#pesteje stevilo oken, ki ustrezajo nekemu hevristicnemu pogoju
def count_windows(grid, num_discs, piece):
    num_windows = 0
    # horizontal
    for row in range(6):
        for col in range(4):
            window = list(grid[row, col:col+4])
            if check_window(window, num_discs, piece):
                num_windows += 1
    # vertical
    for row in range(3):
        for col in range(7):
            window = list(grid[row:row+4, col])
            if check_window(window, num_discs, piece):
                num_windows += 1
    # positive diagonal
    for row in range(3):
        for col in range(4):
            window = list(grid[range(row, row+4), range(col, col+4)])
            if check_window(window, num_discs, piece):
                num_windows += 1
    # negative diagonal
    for row in range(3, 6):
        for col in range(4):
            window = list(grid[range(row, row-4, -1), range(col, col+4)])
            if check_window(window, num_discs, piece):
                num_windows += 1
    return num_windows

class Connect4(ALModule):
    """ A simple module able to react
    to facedetection events

    """
    def __init__(self, name):
        ALModule.__init__(self, name)

        #PROXY-ji
        self.govor = ALProxy("ALTextToSpeech")
        self.foto = ALProxy("ALPhotoCapture")
        self.mem = ALProxy("ALMemory")
        self.pre = ALProxy("ALMotion")
        self.tra = ALProxy("ALTracker")
        self.pos = ALProxy("ALRobotPosture")
        self.lan = ALProxy("ALLandMarkDetection")
        self.pog = ALProxy("ALSpeechRecognition")

        #barva igralca in barva robota
        self.IGRALEC=1
        self.NAO=2

        #vprasamo igralca s katero barvo plosckov bo igral, 
        #ce ga robot slucajno ne razume igra z rdecimi NAO pa z rumenimi
        self.pog.setLanguage("English")
        besede = ["red","yellow"]
        self.pog.setVocabulary(besede, False)
        self.pog.subscribe("govor")

        self.mem.raiseEvent("LastWordRecognized", '')

        self.govor.say("What color of disks will you be playing as? Red or yellow?")

        barva=""
        while barva=="" or barva[0]=='':
            barva=self.mem.getData("LastWordRecognized")
            time.sleep(1)

        self.pog.unsubscribe("govor")

        if barva[0]=='red':
            self.IGRALEC=RDECA
            self.NAO=RUMENA
            self.govor.say("Then I'll be playing as yellow.")
        elif barva[0]=='yellow':
            self.IGRALEC=RUMENA
            self.NAO=RDECA
            self.govor.say("Then I'll be playing as red.")
        else:
            self.IGRALEC=RDECA
            self.NAO=RUMENA
            self.govor.say("I didn't understand you. I'll be playing as yellow you play as red.")

        #Event
        self.mem.subscribeToEvent("ALTactileGesture/Gesture", "Spremenljivka", "naDotik")
    
    def naDotik(self, *_args):
        """ izvede se ko se dotaknemo """
        global KONEC

        #pramakni v pozicijo za slikanje
        self.pre.setStiffnesses("HeadPitch", 1)
        self.pre.angleInterpolation("HeadPitch", 0.4, 1.0, True)

        #Fotografiranje
        image = fot(self.foto)
        if image is None:
            self.govor.say("Something went wrong when opening the image. Try again.")
            self.mem.subscribeToEvent("ALTactileGesture/Gesture", "Spremenljivka", "naDotik")
            return

        #Zaznavanje krogov
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)

        #Tabela tock v centru krogov
        pts = cv2.KeyPoint_convert(keypoints)

        #Preveri ce je prebral 42 krogov, ce ni premakne glavo in slika se enkrat
        if pts==() or pts.size/2!=42:
            print(str(pts.size/2))
            self.pre.angleInterpolation("HeadPitch", 0.3, 1.0, True)

            #Fotografiranje
            image = fot(self.foto)
            if image is None:
                self.govor.say("Something went wrong when opening the image. Try again.")
                self.mem.subscribeToEvent("ALTactileGesture/Gesture", "Spremenljivka", "naDotik")
                return
            
            #Zaznavanje krogov
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(image)

            #Tabela tock v centru krogov
            pts = cv2.KeyPoint_convert(keypoints)

        #Preveri ce je prebral 42 krogov, ce ni premakne glavo in slika se enkrat
        if pts==() or pts.size/2!=42:
            print(str(pts.size/2))
            self.pre.angleInterpolation("HeadPitch", 0.5, 1.0, True)

            #Fotografiranje
            image = fot(self.foto)
            if image is None:
                self.govor.say("Something went wrong when opening the image. Try again.")
                self.mem.subscribeToEvent("ALTactileGesture/Gesture", "Spremenljivka", "naDotik")
                return
            
            #Zaznavanje krogov
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(image)

            #Tabela tock v centru krogov
            pts = cv2.KeyPoint_convert(keypoints)

        #Se enkrat preveri ce je prebral 42 krogov, ce ni pokaze katere je zaznal
        if pts==() or pts.size/2!=42:
            print(str(pts.size/2))
            self.govor.say("Something went wrong when reading the board. Try to move me into a better position.")
            im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Keypoints", im_with_keypoints)
            cv2.waitKey(0)
            self.mem.subscribeToEvent("ALTactileGesture/Gesture", "Spremenljivka", "naDotik")
            return

        #Urejanje tabele
        pts.view('float32, float32').sort(order=['f1'], axis=0)
        pts =np.split(pts, 6)
        for s in range(6):
            pts[s].view('float32, float32').sort(order=['f0'], axis=0)

        #Ugotavljanje barv
        board=[[0]*7 for p in range(6)]

        for i in range(6):
            for j in range(7):

                #Koordinati centra kroga
                x=int(round(pts[i][j][0]))
                y=int(round(pts[i][j][1]))

                #Racunanje povprecja RGB vrednosti pikslov okrog koordinat
                r=0
                g=0
                b=0
                for o in range(8):
                    for u in range(8):
                        r=r+image[y+o-2, x+u-2, 2]
                        g=g+image[y+o-2, x+u-2, 1]
                        b=b+image[y+o-2, x+u-2, 0]
                r=r/64
                g=g/64
                b=b/64

                #Ugotavljanje barve
                if (4*b<g and 4*b<r) and (g>40 and r>40):
                    board[i][j]=RUMENA
                    #print("rumen")
                if g+b < r/3:
                    board[i][j]=RDECA
                    #print("rdec")
        
        #preverimo ce je clovek zmagal
        if(winning_move(board, self.IGRALEC)):
            self.govor.say("You won, congratulations.")
            self.govor.say("End of the game, shutting down")
            #Izbrisi sliko
            os.remove("image.jpg")
            self.pre.rest()
            KONEC=True
            return

        #preverimo ce obstaja mozna poteza
        if not(can_move(board)):
            self.govor.say("Game ended in a draw, good game.")
            self.govor.say("Shutting down")
            #Izbrisi sliko
            os.remove("image.jpg")
            self.pre.rest()
            KONEC=True
            return

        #Robot izbere potezo
        col = move(board, self.NAO)
        for a in range(6):
            if board[5-a][col]==0:
                board[5-a][col]=self.NAO
                break

        #Robot pove v kateri stolpec bo vrgel ploscek
        reci="Throwing into column number "+str(col+1)
        self.govor.say(reci)

        #Izbrisi sliko
        os.remove("image.jpg")

        #Ponovno povezemo na event
        self.mem.subscribeToEvent("ALTactileGesture/Gesture", "Spremenljivka", "naDotik")

def main():
    """ Main entry point
    """
    parser = OptionParser()
    parser.add_option("--pip",
        help="Parent broker port. The IP address or your robot",
        dest="pip")
    parser.add_option("--pport",
        help="Parent broker port. The port NAOqi is listening to",
        dest="pport",
        type="int")
    parser.set_defaults(
        pip=NAO_IP,
        pport=9559)

    (opts, args_) = parser.parse_args()
    pip   = opts.pip
    pport = opts.pport

    # We need this broker to be able to construct
    # NAOqi modules and subscribe to other modules
    # The broker must stay alive until the program exists
    myBroker = ALBroker("myBroker",
       "0.0.0.0",   # listen to anyone
       0,           # find a free port and use it
       pip,         # parent broker IP
       pport)       # parent broker port


    # Warning: HumanGreeter must be a global variable
    # The name given to the constructor must be the name of the
    # variable
    global Spremenljivka
    Spremenljivka = Connect4("Spremenljivka")

    try:
        while True:
            if KONEC:
                myBroker.shutdown()
                sys.exit(0) 
            time.sleep(1)
    except KeyboardInterrupt:
        print
        print("Interrupted by user, shutting down")
        myBroker.shutdown()
        sys.exit(0)
        
if __name__ == "__main__":
    main()
