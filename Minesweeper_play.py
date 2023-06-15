#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install keras')


# In[ ]:


#get_ipython().system('pip install tensorflow')


# In[ ]:


#pip install gym


# In[2]:


import keras
from keras.layers import Dense, Activation ,Conv2D, Flatten,Reshape 
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np


# In[3]:


class Cell:
    def __init__(self, content, x, y):
        self.content = content
        self.flagged = False
        self.open = False
        self.x = x
        self.y = y
    
    #reveal a cell (open it)
    def reveal(self):
        if self.open:
            print(f"cell ({self.x},{self.y}) is already open")
            return
        if self.flagged:
            print(f"cell ({self.x},{self.y}) is already flagged")
            return
        self.open = True
    
    #flag a cell
    def flag(self):
        if self.flagged:
            print(f"cell ({self.x},{self.y}) is already flagged")
            return
        if self.open:
            print(f"cell ({self.x},{self.y}) cannot be flagged because it is already open")
            return
        self.flagged = True

    def getContent(self):
      if self.open==True:
        return self.content
      else:
        return -2


# In[4]:


import random

class MinesweeperMap:
    def __init__(self, difficulty): #map generation
        difficulty = difficulty.lower()
        if difficulty == "hard":
            self.numberOfBombs = 85   
            self.map = [[Cell(0, r, c) for c in range(30)] for r in range(16)]
        elif difficulty == "medium":
            self.numberOfBombs = 40  
            self.map = [[Cell(0, r, c) for c in range(16)] for r in range(16)]
        else: # easy level
            self.numberOfBombs = 10
            self.map = [[Cell(0, r, c) for c in range(9)] for r in range(9)]
            
        
        cords = [ [r, c] for r in range(len(self.map)) for c in range(len(self.map[0]))  ]
        

        cords.pop(0) # removing the first cords (0,0) so it can't be a bomb
        cords.pop(1)
        cords.pop(len(self.map[0]))
        
        # setting up the locations of the bomb cells randomly
        for i in range(self.numberOfBombs):
            rand = random.randint(0, len(cords)-1)
            randArr = cords[rand]
            cords.pop(rand)
            self.map[randArr[0]][randArr[1]] = Cell(-1, randArr[0], randArr[1])
        
        for r in range(len(self.map)):
            for c in range(len(self.map[0])):
                if self.map[r][c].content != -1:
                    n = self.numberOfSurroundingBombs(r, c)
                    self.map[r][c] = Cell(n, r, c)
    

    #only to generate the map (not used in SolverCSP)
    def numberOfSurroundingBombs(self, x, y):
        #this is a useless cell just to pass the x,y coords
        trashCell = Cell(9999, x, y)
        #get the neighbors of the cell
        neighbors = self.getNeighbors(trashCell)
        n = 0
    # count the number of bombs among the neighbors
        for cell in neighbors:
              if cell is not None and cell.content == -1:
                n = n + 1
        return n

    # returns the surrounding neighboring cells of a cell.
    # a cell can have between 3, 5, or 8 neighbors, depending on its location on the map.
    def getNeighbors(self, c):
        neighbors = []
        x, y = c.x, c.y
        if x+1 < len(self.map): # x+1, y
            neighbors.append(self.map[x+1][y])
        if x-1 >= 0: # x-1, y
            neighbors.append(self.map[x-1][y])
        if y-1 >= 0: # x, y-1
            neighbors.append(self.map[x][y-1])
        if y+1 < len(self.map[0]): # x, y+1
            neighbors.append(self.map[x][y+1])
        if x+1 < len(self.map) and y+1 < len(self.map[0]): # x+1, y+1
            neighbors.append(self.map[x+1][y+1])
        if x-1 >= 0 and y-1 >= 0: # x-1, y-1
            neighbors.append(self.map[x-1][y-1])
        if x+1 < len(self.map) and y-1 >= 0: # x+1, y-1
            neighbors.append(self.map[x+1][y-1])
        if x-1 >= 0 and y+1 < len(self.map[0]): # x-1, y+1
            neighbors.append(self.map[x-1][y+1])
        return neighbors

    #prints the current state of the map
    def drawMap(self):
      for i in range(len(self.map)):
          for l in range(len(self.map[0])):
            print("----", end="")
          print("---")
          for j in range(len(self.map[0])):
              if self.map[i][j].open:
                if (self.map[i][j].content==-1):
                  print(" | B", end="")
                else:
                  print(" | " + str(self.map[i][j].content), end="")
              else:
                if self.map[i][j].flagged:
                    print(" | X", end="")
                else:
                    print(" |  ", end="")
                #print(" | " +str(self.map[i][j].content), end="")  #to print the uncovered version of the map
          print(" | ")
      for l in range(len(self.map[0])):
        print("----", end="")
      print("---")
    
    
    # the functions below functions are made for the SolverCSP

    # return a list of the surrounding flags of a cell
    def surroundingFlags(self, c):
      list = self.getNeighbors(c)
      bombedCells = []
      for cell in list:
        if cell.flagged:
          bombedCells.append(cell)
      return bombedCells

    def numberOfSurroundingFlags(self, c):
      return len(self.surroundingFlags(c))

    # returns a list of the surrounding unexplored cells of a cell (unexplored means not flagged and not opened)
    def surroundingUnexplored(self, c):
      list = self.getNeighbors(c)
      unexploredCells = []
      for cell in list:
        if not cell.flagged and not cell.open:
          unexploredCells.append(cell)
      return unexploredCells

    def numberOfSurroundingUnexplored(self, c):
      return len(self.surroundingUnexplored(c))

    # return the number of flagged cells in the whole map
    def getNumberOfFlags(self):
      n = 0
      for i in range(len(self.map)):
        for j in range(len(self.map[0])):
          if self.map[i][j].flagged:
            #print("flagged")
            n = n + 1
          #else:
            #print("not")
      #print("done the loop")
      return n

    # returns the number of explored cells in the whole map (explored means either flagged or opened)
    def getNumberOfExplored(self):
      n = 0
      for i in range(len(self.map)):
        for j in range(len(self.map[0])):
          if self.map[i][j].flagged or self.map[i][j].open:
            n = n+ 1
      return n

    #returns a list of all unexplored cells in the whole map
    def getUnexploredCells(self):
        arr = []
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if not self.map[i][j].flagged and not self.map[i][j].open:
                    arr.append(self.map[i][j])
        return arr
    
    #returns whether a cell has any unexplored neighbors
    def hasUnexploredNeighbor(self, c):
        neighbors = self.getNeighbors(c)
        for cell in neighbors:
            if not cell.flagged and not cell.open:
                return True
        return False

    #returns whether a cell has any explored neighbors
    def hasExploredNeighbor(self, c):
        neighbors = self.getNeighbors(c)
        for cell in neighbors:
            if ((not cell.flagged) and (cell.open)):
                return True
        return False
    
    #return true if a cell is open and has unexplored neighbors
    def openAndHasUnexploredNeighbors(self, c):
        if c.open and self.hasUnexploredNeighbor(c):
            return True
        return False
    
    #returns a list of open neighbors of a cell, and at least one of those open neighbors still have at least one unexplored neighboring cell
    def openNeighborsWithUnexploredNeighbors(self, c):
        neighbors = self.getNeighbors(c)
        wantedNeighbors = []
        for cell in neighbors:
            if self.hasUnexploredNeighbor(cell) and cell.open and not cell.flagged:
                wantedNeighbors.append(cell)
        return wantedNeighbors

     # returns true if a cell has any open neighbors and at least one of those neighbors have at least one unexplored neighbor.
    def hasOpenNeighborsWithUnexploredNeighbors(self, c):
        neighbors = self.getNeighbors(c)
        for cell in neighbors:
            if self.hasUnexploredNeighbor(cell) and cell.open and not cell.flagged:
                return True
        return False
    
    # get the flagged cells of a cell and returns their surrounding neighbors that have at least one unexplored cell.
    def flaggedNeighborsWithUnexploredNeighbors(self, c):
        neighbors = self.getNeighbors(c)
        wantedNeighbors = []
        for cell in neighbors:
            if self.hasOpenNeighborsWithUnexploredNeighbors(cell) and cell.flagged and not cell.open:
                arrr = self.openNeighborsWithUnexploredNeighbors(cell)
                for i in range(len(arrr)):
                    wantedNeighbors.append(arrr[i])
        return wantedNeighbors
    
    # returns a random not opened and not flagged cell from the map
    def getRandomCell(self):
        undiscovered = []
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if (not self.map[i][j].flagged) and (not self.map[i][j].open):
                    undiscovered.append(self.map[i][j])
        if(len(undiscovered)>0):
            rand = random.randint(0, len(undiscovered)-1)
            return undiscovered[rand]
        else:
            return -1
    
    #checks whether a cell has at least 3 explored neighbors
    def hasAtLeastThreeExploredNeighbors(self, c):
        neighbors = self.getNeighbors(c)
        count = 0
        for cell in neighbors:
            if ((cell.flagged) or (cell.open)):
                count = count + 1
                if(count==3):
                    return True
        return False
    
    #returns a baord of 5x5 for RL
    def returnsFiveBoard(self):
      if(len(self.map[0])==9):
        take = [2,3,4,5,6]
      elif(len(self.map)==16):
        take = [2,3,4,5,6,7,8,9,10,11,12,13]
      else:
        take = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
      smallBoards = []
      for i in take:
          for j in take:
              smallBoard = []
              if self.map[i][j].open==False and self.map[i][j].flagged==False and self.hasAtLeastThreeExploredNeighbors(self.map[i][j]):
                  for m in [i-2,i-1,i,i+1,i+2]:
                      smallBoardRow = []
                      for n in [j-2,j-1,j,j+1,j+2]:
                          smallBoardRow.append(self.map[m][n])
                      smallBoard.append(smallBoardRow)
                  smallBoards.append(smallBoard)
      return smallBoards

    


# In[5]:


# this is you are importing the model from Drive
# import os
# os.chdir('/content/drive/My Drive/Colab Notebooks/RL Minesweeper')

agent = load_model('best.h5')

def solve_by_RL(state):
  obs = state
  observation = np.array([[obs[r][c].getContent() for c in range(len(obs[0]))] for r in range(len(obs))])
  #print(observation)
  observation = observation.flatten()
  observation = observation[np.newaxis, :]
  actions = agent.predict(observation)
  action = np.argmax(actions)
  return action


# In[6]:




#for RL

import copy

class SolverCSP:
    countRLRightDecisions = 0
    countRLWrongDecisions = 0
    countWentToRL = 0
    percentageOfSolving = 0
    my_list = []
    def solveMinesweeper(map: MinesweeperMap):
        stack = []
        map.map[0][0].reveal()
        SolverCSP.my_list.append(map.map[0][0])
        #map.drawMap()
        stack.append(map.map[0][0])
       
        while (map.numberOfBombs != map.getNumberOfFlags()):
            while len(stack)!=0:
                stack = SolverCSP.firstLevel(map, stack)
                
            stack = SolverCSP.secondLevel(map, stack)
            if len(stack)==0:
                #stack = SolverCSP.thirdLevel(map, stack)
                if len(stack)==0:
                  brds = map.returnsFiveBoard()
                  while(len(stack)==0):
                    if(len(brds)==0):
                        if map.numberOfBombs == map.getNumberOfFlags():
                            return True
                        else:
                          if ((len(map.map) * len(map.map[0])) - map.getNumberOfExplored()) == map.numberOfBombs - map.getNumberOfFlags():
                            arr = map.getUnexploredCells()
                            for i in range(len(arr)):
                                map.map[arr[i].x][arr[i].y].reveal()
                                SolverCSP.my_list.append(arr[i])
                            return True
                          c = map.getRandomCell()
                          if(c==-1):
                            return True
                          #print("RANDOMLY")
                          if c.content == -1:
                              map.map[c.x][c.y].reveal()
                              SolverCSP.my_list.append(c)
                              #map.drawMap()
                              SolverCSP.percentageOfSolving = map.getNumberOfExplored()
                              return False
                          else:
                              map.map[c.x][c.y].reveal()
                              stack.append(map.map[c.x][c.y])
                              SolverCSP.my_list.append(c)
                              #map.drawMap()
                              openNeighborsWithUnexploredNeighbors = map.openNeighborsWithUnexploredNeighbors(map.map[c.x][c.y])
                              for i in range(len(openNeighborsWithUnexploredNeighbors)):
                                  stack.append(map.map[openNeighborsWithUnexploredNeighbors[i].x][openNeighborsWithUnexploredNeighbors[i].y])
                    else:
                        brd=brds[0]
                        brds.pop(0)
                        decision = solve_by_RL(brd)
                        cl = brd[2][2]
                        SolverCSP.countWentToRL = SolverCSP.countWentToRL + 1
                        if(decision==1):
                          if(map.map[cl.x][cl.y].content==-1):
                            SolverCSP.countRLWrongDecisions = SolverCSP.countRLWrongDecisions + 1
                            map.map[cl.x][cl.y].reveal()
                            SolverCSP.my_list.append(cl)
                            #map.drawMap()
                            SolverCSP.percentageOfSolving = map.getNumberOfExplored()
                            return False
                          map.map[cl.x][cl.y].reveal()
                          SolverCSP.my_list.append(cl)
                          SolverCSP.countRLRightDecisions = SolverCSP.countRLRightDecisions + 1
                          stack.append(map.map[cl.x][cl.y])
                          #map.drawMap()
                          openNeighborsWithUnexploredNeighbors = map.openNeighborsWithUnexploredNeighbors(map.map[cl.x][cl.y])
                          for i in range(len(openNeighborsWithUnexploredNeighbors)):
                              stack.append(map.map[openNeighborsWithUnexploredNeighbors[i].x][openNeighborsWithUnexploredNeighbors[i].y])
        return True
    

    def firstLevel(map: MinesweeperMap, stack):
      top = stack.pop()
      if top.content == map.numberOfSurroundingUnexplored(top) + map.numberOfSurroundingFlags(top):
        unexploredNeighbors = map.surroundingUnexplored(top)
        for neighbor in unexploredNeighbors:
            map.map[neighbor.x][neighbor.y].flag()
            SolverCSP.my_list.append(neighbor)
            #map.drawMap()
        flaggedNeighborsWithUnexploredNeighbors = map.flaggedNeighborsWithUnexploredNeighbors(top)
        for flaggedNeighbor in flaggedNeighborsWithUnexploredNeighbors:
            stack.append(map.map[flaggedNeighbor.x][flaggedNeighbor.y])
      if top.content == map.numberOfSurroundingFlags(top):
        unexploredNeighbors = map.surroundingUnexplored(top)
        for neighbor in unexploredNeighbors:
            map.map[neighbor.x][neighbor.y].reveal()
            SolverCSP.my_list.append(neighbor)
            #map.drawMap()
            stack.append(map.map[neighbor.x][neighbor.y])
        openNeighborsWithUnexploredNeighbors = map.openNeighborsWithUnexploredNeighbors(top)
        for openNeighbor in openNeighborsWithUnexploredNeighbors:
            stack.append(map.map[openNeighbor.x][openNeighbor.y])
      return stack


    def secondLevel(map: MinesweeperMap, stack):
      for i in range(len(map.map)):
        for j in range(len(map.map[0])):
            if map.openAndHasUnexploredNeighbors(map.map[i][j]) and map.hasOpenNeighborsWithUnexploredNeighbors(map.map[i][j]):
                neighb = map.openNeighborsWithUnexploredNeighbors(map.map[i][j])
                for k in range(len(neighb)):
                    first = map.surroundingUnexplored(map.map[i][j])
                    firstContent = map.map[i][j].content
                    second = map.surroundingUnexplored(neighb[k])
                    secondContent = neighb[k].content
                    l = 0
                    while l < len(first):
                        q = 0
                        while q < len(second):
                            if first[l].x== second[q].x and first[l].y == second[q].y:
                                first.pop(l)
                                second.pop(q)
                                l = l-1
                                break
                            q = q+1
                        l =l+1

                    if len(first)==0 and len(second)!=0:
                        res = secondContent - firstContent - map.numberOfSurroundingFlags(neighb[k]) + map.numberOfSurroundingFlags(map.map[i][j])
                        if res == 0:
                            for p in range(len(second)):
                                map.map[second[p].x][second[p].y].reveal()
                                SolverCSP.my_list.append(second[p])
                                #map.drawMap()
                                stack.append(map.map[second[p].x][second[p].y])
                                openNeighborsWithUnexploredNeighbors = map.openNeighborsWithUnexploredNeighbors(second[p])
                                for h in range(len(openNeighborsWithUnexploredNeighbors)):
                                    stack.append(map.map[openNeighborsWithUnexploredNeighbors[h].x][openNeighborsWithUnexploredNeighbors[h].y])
                            return stack
                        if (len(second) + map.numberOfSurroundingFlags(neighb[k]) + firstContent - map.numberOfSurroundingFlags(map.map[i][j])) == secondContent:
                            for p in range(len(second)):
                                map.map[second[p].x][second[p].y].flag()
                                SolverCSP.my_list.append(second[p])
                                #map.drawMap()
                                stack.append(map.map[second[p].x][second[p].y])
                                flaggedNeighborsWithUnexploredNeighbors = map.flaggedNeighborsWithUnexploredNeighbors(second[p])
                                for h in range(len(flaggedNeighborsWithUnexploredNeighbors)):
                                    stack.append(map.map[flaggedNeighborsWithUnexploredNeighbors[h].x][flaggedNeighborsWithUnexploredNeighbors[h].y])
                            return stack
                    if len(second)==0 and len(first)!=0:
                        res = firstContent - secondContent - map.numberOfSurroundingFlags(map.map[i][j]) + map.numberOfSurroundingFlags(neighb[k])
                        if res == 0:
                            for p in range(len(first)):
                                map.map[first[p].x][first[p].y].reveal()
                                SolverCSP.my_list.append(first[p])
                                #map.drawMap()
                                stack.append(map.map[first[p].x][first[p].y])
                                openNeighborsWithUnexploredNeighbors = map.openNeighborsWithUnexploredNeighbors(first[p])
                                for h in range(len(openNeighborsWithUnexploredNeighbors)):
                                    stack.append(map.map[openNeighborsWithUnexploredNeighbors[h].x][openNeighborsWithUnexploredNeighbors[h].y])
                            return stack
                        if (len(first) + map.numberOfSurroundingFlags(map.map[i][j]) + secondContent - map.numberOfSurroundingFlags(neighb[k])) == firstContent:
                            for p in range(len(first)):
                                map.map[first[p].x][first[p].y].flag()
                                SolverCSP.my_list.append(first[p])
                                #map.drawMap()
                                stack.append(map.map[first[p].x][first[p].y])
                                flaggedNeighborsWithUnexploredNeighbors = map.flaggedNeighborsWithUnexploredNeighbors(first[p])
                                for h in range(len(flaggedNeighborsWithUnexploredNeighbors)):
                                    stack.append(map.map[flaggedNeighborsWithUnexploredNeighbors[h].x][flaggedNeighborsWithUnexploredNeighbors[h].y])
                            return stack

                    if len(second)==1 and len(first)==1:
                        res = secondContent - firstContent - map.numberOfSurroundingFlags(neighb[k]) + map.numberOfSurroundingFlags(map.map[i][j])
                        if res==-1:
                          map.map[second[0].x][second[0].y].reveal()
                          SolverCSP.my_list.append(second[0])
                          #map.drawMap()
                          map.map[first[0].x][first[0].y].flag()
                          SolverCSP.my_list.append(first[0])
                          #map.drawMap()
                          stack.append(map.map[second[0].x][second[0].y])
                          stack.append(map.map[first[0].x][first[0].y])
                          openNeighborsWithUnexploredNeighbors = map.openNeighborsWithUnexploredNeighbors(second[0])
                          for h in range(len(openNeighborsWithUnexploredNeighbors)):
                              stack.append(map.map[openNeighborsWithUnexploredNeighbors[h].x][openNeighborsWithUnexploredNeighbors[h].y])
                          flaggedNeighborsWithUnexploredNeighbors = map.flaggedNeighborsWithUnexploredNeighbors(first[0])
                          for t in range(len(flaggedNeighborsWithUnexploredNeighbors)):
                              stack.append(map.map[flaggedNeighborsWithUnexploredNeighbors[t].x][flaggedNeighborsWithUnexploredNeighbors[t].y])
                          return stack
                        if res==1:
                          map.map[second[0].x][second[0].y].flag()
                          SolverCSP.my_list.append(second[0])
                          #map.drawMap()
                          map.map[first[0].x][first[0].y].reveal()
                          SolverCSP.my_list.append(first[0])
                          #map.drawMap()
                          stack.append(map.map[second[0].x][second[0].y])
                          stack.append(map.map[first[0].x][first[0].y])
                          openNeighborsWithUnexploredNeighbors = map.openNeighborsWithUnexploredNeighbors(first[0])
                          for h in range(len(openNeighborsWithUnexploredNeighbors)):
                              stack.append(map.map[openNeighborsWithUnexploredNeighbors[h].x][openNeighborsWithUnexploredNeighbors[h].y])
                          flaggedNeighborsWithUnexploredNeighbors = map.flaggedNeighborsWithUnexploredNeighbors(second[0])
                          for t in range(len(flaggedNeighborsWithUnexploredNeighbors)):
                              stack.append(map.map[flaggedNeighborsWithUnexploredNeighbors[t].x][flaggedNeighborsWithUnexploredNeighbors[t].y])
                          return stack
      return stack


    def thirdLevel(map: MinesweeperMap, stack):
       map1 = copy.deepcopy(map)
       stack1 = []
       for i in range(len(map1.map)):
        for j in range(len(map1.map[0])):
            cell = map1.map[i][j]
            if not cell.open and not cell.flagged and map1.hasExploredNeighbor(cell):
              cell.flag()
              flaggedNeighborsWithUnexploredNeighbors = map1.flaggedNeighborsWithUnexploredNeighbors(cell)
              for h in range(len(flaggedNeighborsWithUnexploredNeighbors)):
                stack1.append(map1.map[flaggedNeighborsWithUnexploredNeighbors[h].x][flaggedNeighborsWithUnexploredNeighbors[h].y])
              while len(stack1)!=0:
                stack1 = SolverCSP.firstLevelMod(map1, stack1)
              res = SolverCSP.checkMistakes(map1)
              if(res):
                #print("USED 3")
                #print("i=" + str(i) + " j=" + str(j))
                cell.flagged=False
                map.map[i][j].reveal()
                #map.drawMap()
                stack.append(map.map[i][j])
                openNeighborsWithUnexploredNeighbors = map.openNeighborsWithUnexploredNeighbors(map.map[i][j])
                for h in range(len(openNeighborsWithUnexploredNeighbors)):
                  stack.append(map.map[openNeighborsWithUnexploredNeighbors[h].x][openNeighborsWithUnexploredNeighbors[h].y])
                return stack
              else:
                cell.flagged=False
       return stack


    def firstLevelMod(map1: MinesweeperMap, stack1):
      noList = []
      top = stack1.pop()
      if top.content == map1.numberOfSurroundingUnexplored(top) + map1.numberOfSurroundingFlags(top):
        unexploredNeighbors = map1.surroundingUnexplored(top)
        for neighbor in unexploredNeighbors:
            map1.map[neighbor.x][neighbor.y].flag()
            #map.drawMap()
        flaggedNeighborsWithUnexploredNeighbors = map1.flaggedNeighborsWithUnexploredNeighbors(top)
        for flaggedNeighbor in flaggedNeighborsWithUnexploredNeighbors:
            stack1.append(map1.map[flaggedNeighbor.x][flaggedNeighbor.y])
      if top.content == map1.numberOfSurroundingFlags(top):
        unexploredNeighbors = map1.surroundingUnexplored(top)
        for neighbor in unexploredNeighbors:
            map1.map[neighbor.x][neighbor.y].reveal()
            noList.append([neighbor.x,neighbor.y])
            #map.drawMap()
        openNeighborsWithUnexploredNeighbors = map1.openNeighborsWithUnexploredNeighbors(top)
        for openNeighbor in openNeighborsWithUnexploredNeighbors:
            enter= True
            for nl in noList:
              if (nl[0]==openNeighbor.x and nl[1]==openNeighbor.y):
                enter=False
                break
            if(enter):    
              stack1.append(map1.map[openNeighbor.x][openNeighbor.y])
      return stack1

    def checkMistakes(map: MinesweeperMap):
      for i in range(len(map.map)):
        for j in range(len(map.map[0])):
          if(map.map[i][j].open and map.map[i][j].content!=-1):
            if map.map[i][j].content > map.numberOfSurroundingUnexplored(map.map[i][j]) + map.numberOfSurroundingFlags(map.map[i][j]):
              return True
            if map.map[i][j].content < map.numberOfSurroundingFlags(map.map[i][j]):
              return True
      return False



        


# In[ ]:


import pygame
import time

# Initialize Pygame
pygame.init()

# Set the display size
display_width = 700
display_height = 700

# Set the color for the background
bg_color = (255, 255, 255)

# Set the font for the text
font = pygame.font.SysFont('Calibri', 30)

# Create a function to draw the board
def draw_board(board):
    screen.fill(bg_color)
    cell_size = display_width // len(board)
    cell_border = 2

    for i in range(len(board)):
        for j in range(len(board[0])):
            # Set the cell color based on the value
            if board[i][j] == -1:
                cell_color = (255, 0, 0)
            elif board[i][j] == 0:
                cell_color = (172, 172, 172)
            elif board[i][j] == 100:
                cell_color = (255, 255, 255)
            elif board[i][j] == 50:
                cell_color = (0, 255, 0)
            else:
                cell_color = (0, 0, 255)

            cell_rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, cell_color, cell_rect)
            pygame.draw.rect(screen, (0, 0, 0), cell_rect, cell_border)

            if board[i][j] > 0 and board[i][j]<50:
                text = font.render(str(board[i][j]), True, (0, 0, 0))
                text_rect = text.get_rect(center=cell_rect.center)
                screen.blit(text, text_rect)

    pygame.display.update()


screen = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Minesweeper')


map = MinesweeperMap("medium")
solve = SolverCSP.solveMinesweeper(map)
SolverCSP.solveMinesweeper(map)
openingList = SolverCSP.my_list

board = [[100 for j in range(len(map.map[0]))] for i in range(len(map.map))]
draw_board(board)
time.sleep(0.1)

i=0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
          

    if(i<len(openingList)):
      if(openingList[i].flagged):
        board[openingList[i].x][openingList[i].y] = 50
        i=i+1
      elif (openingList[i].content==-1):
        board[openingList[i].x][openingList[i].y] = openingList[i].content
        i=len(openingList)
      else:
        board[openingList[i].x][openingList[i].y] = openingList[i].content
        i = i+1
       
    draw_board(board)
    time.sleep(0.1)

