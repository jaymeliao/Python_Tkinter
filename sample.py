import tkinter as tk
"""
Window
    -create a new window : Tk()
    -set window size: geometry("600x300")
    -set window color to black: configure(bg="black")
    -set window title: title("<app name>")
Label:
    -create a text label：Label(text="", foreground="white", background="red")
    -Add to window: pack()  --> pack() first then you can position the widget
    -Position a text label: place(x=,y=)
  

Text Input:
    -create a text input: Entry()
    -Add to window: pack()
    -Position a text input: place(x=,y=)

Button:
    -create a button: Button(text="click me")
    -Add to window: pack()
    -Position a button: place(x=,y=)


ComboBox:

CheckButton:

Radio:

ScrolledText:

SpinBox:

Menu Bar:

Canvas: provide ability to draw shapes(lines, rectangle, polygon, ovals..); can contain image or bitmaps


 
"""



"""
Mouse Events
	<Button-1>   -left mouse button
        <Button-2>   -middle mouse button
	<Button-3>   -right mouse button
    
        <ButtonRelease-1>   -left mouse button release
        <Double-Button-1>   -double click on left mouse button 
        <Enter>             -mouse pointer entered widget
        <Leave>             -mouse pointer leave the  widget   

KeyBoard Events
        <Return>            -enter key press
        <Key>               -a key pressed



"""

win = tk.Tk()
win.configure(bg="black")
win.title("Hacker Terminal")
win.geometry("800x900")
text=tk.Label(text="404 not found", foreground="white", background="red",font=('Arial',60))
text.pack()
text.place(x=0,y=0)
win.mainloop()

