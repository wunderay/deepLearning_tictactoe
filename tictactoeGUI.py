# Tic-Tac-Toe GUI for Deep Learning Project
from tkinter import *
from tkinter import messagebox

root = Tk()
root.title('Tic-Tac-Toe - Deep Learning')
root.resizable(False, False)


def gameCheck():
    # Go trough game conditions to see if game has ended
    global winner
    winner = False

    # Check to see if player "X" wins
    if(s1["text"] == "X" and s2["text"] == "X" and s3["text"] == "X"):
        s1.config(bg="gold")
        s2.config(bg="gold")
        s3.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "X wins!")
        setupGame()
    elif(s4["text"] == "X" and s5["text"] == "X" and s6["text"] == "X"):
        s4.config(bg="gold")
        s5.config(bg="gold")
        s6.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "X wins!")
        setupGame()
    elif(s7["text"] == "X" and s8["text"] == "X" and s9["text"] == "X"):
        s7.config(bg="gold")
        s8.config(bg="gold")
        s9.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "X wins!")
        setupGame()
    elif(s1["text"] == "X" and s4["text"] == "X" and s7["text"] == "X"):
        s1.config(bg="gold")
        s4.config(bg="gold")
        s7.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "X wins!")
        setupGame()
    elif(s2["text"] == "X" and s5["text"] == "X" and s8["text"] == "X"):
        s2.config(bg="gold")
        s5.config(bg="gold")
        s8.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "X wins!")
        setupGame()
    elif(s3["text"] == "X" and s6["text"] == "X" and s9["text"] == "X"):
        s3.config(bg="gold")
        s6.config(bg="gold")
        s9.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "X wins!")
        setupGame()
    elif(s1["text"] == "X" and s5["text"] == "X" and s9["text"] == "X"):
        s1.config(bg="gold")
        s5.config(bg="gold")
        s9.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "X wins!")
        setupGame()
    elif(s3["text"] == "X" and s5["text"] == "X" and s7["text"] == "X"):
        s3.config(bg="gold")
        s5.config(bg="gold")
        s7.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "X wins!")
        setupGame()
    # Check to see if player "O" wins
    elif(s1["text"] == "O" and s2["text"] == "O" and s3["text"] == "O"):
        s1.config(bg="gold")
        s2.config(bg="gold")
        s3.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "O wins!")
        setupGame()
    elif(s4["text"] == "O" and s5["text"] == "O" and s6["text"] == "O"):
        s4.config(bg="gold")
        s5.config(bg="gold")
        s6.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "O wins!")
        setupGame()
    elif(s7["text"] == "O" and s8["text"] == "O" and s9["text"] == "O"):
        s7.config(bg="gold")
        s8.config(bg="gold")
        s9.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "O wins!")
        setupGame()
    elif(s1["text"] == "O" and s4["text"] == "O" and s7["text"] == "O"):
        s1.config(bg="gold")
        s4.config(bg="gold")
        s7.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "O wins!")
        setupGame()
    elif(s2["text"] == "O" and s5["text"] == "O" and s8["text"] == "O"):
        s2.config(bg="gold")
        s5.config(bg="gold")
        s8.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "O wins!")
        setupGame()
    elif(s3["text"] == "O" and s6["text"] == "O" and s9["text"] == "O"):
        s3.config(bg="gold")
        s6.config(bg="gold")
        s9.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "O wins!")
        setupGame()
    elif(s1["text"] == "O" and s5["text"] == "O" and s9["text"] == "O"):
        s1.config(bg="gold")
        s5.config(bg="gold")
        s9.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "O wins!")
        setupGame()
    elif(s3["text"] == "O" and s5["text"] == "O" and s7["text"] == "O"):
        s3.config(bg="gold")
        s5.config(bg="gold")
        s7.config(bg="gold")
        winner = True
        messagebox.showinfo("Tic-Tac-Toe", "O wins!")
        setupGame()
    if(count == 9 and winner == False):
        messagebox.showinfo("Tic-Tac-Toe", "Tie!")
        setupGame()


def takeTurn(s):
    # Function to determine if button has been clicked
    global turnToggle, count
    if(s["text"] == " " and turnToggle == True):
        s["text"] = "X"
        turnToggle = False
        count += 1
        gameCheck()
    elif(s["text"] == " " and turnToggle == False):
        s["text"] = "O"
        turnToggle = True
        count += 1
        gameCheck()
    else:
        messagebox.showerror("Tic-Tac-Toe", "Square unavailable")


def setupGame():
    # Setup a clean board; used as function to reset board after a game
    global s1, s2, s3, s4, s5, s6, s7, s8, s9
    global turnToggle, count
    turnToggle = True
    count = 0
    s1 = Button(root, text=" ", height=10, width=20,
                bg="SystemButtonFace", command=lambda: takeTurn(s1))
    s2 = Button(root, text=" ", height=10, width=20,
                bg="SystemButtonFace", command=lambda: takeTurn(s2))
    s3 = Button(root, text=" ", height=10, width=20,
                bg="SystemButtonFace", command=lambda: takeTurn(s3))
    s4 = Button(root, text=" ", height=10, width=20,
                bg="SystemButtonFace", command=lambda: takeTurn(s4))
    s5 = Button(root, text=" ", height=10, width=20,
                bg="SystemButtonFace", command=lambda: takeTurn(s5))
    s6 = Button(root, text=" ", height=10, width=20,
                bg="SystemButtonFace", command=lambda: takeTurn(s6))
    s7 = Button(root, text=" ", height=10, width=20,
                bg="SystemButtonFace", command=lambda: takeTurn(s7))
    s8 = Button(root, text=" ", height=10, width=20,
                bg="SystemButtonFace", command=lambda: takeTurn(s8))
    s9 = Button(root, text=" ", height=10, width=20,
                bg="SystemButtonFace", command=lambda: takeTurn(s9))

    # Plot squares to screen
    s1.grid(row=0, column=0)
    s2.grid(row=0, column=1)
    s3.grid(row=0, column=2)
    s4.grid(row=1, column=0)
    s5.grid(row=1, column=1)
    s6.grid(row=1, column=2)
    s7.grid(row=2, column=0)
    s8.grid(row=2, column=1)
    s9.grid(row=2, column=2)


setupGame()
root.mainloop()
