##############################################################################
# RNN Chatbot trained with the USA subgroup of the Twitter data we datamined #
##############################################################################

import tensorflow as tf
from tkinter import *

def getResponse(prompt):
  states = None
  next_char = tf.constant([prompt])
  result = ['']

  for n in range(0, 280+1):
    next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
    result.append(next_char)

  result = tf.strings.join(result)[0].numpy().decode("utf-8")
  return result

def send():
    print("send() called.")
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = getResponse(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# Load chatbot model
one_step_reloaded = tf.saved_model.load('./one-step/')

base = Tk()
base.title("Fwitter")
base.geometry("400x500")
base.resizable(width=False, height=False)

# Create chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

# Bind scrollbar to chat windows
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create button to send messages
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height="5", bd=0, bg="#32de97", activebackground="#3c9d9b", fg="#ffffff", command=send)

# Create the box to enter messages
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()