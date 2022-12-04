import tensorflow as tf

one_step_reloaded = tf.saved_model.load('./one-step/')

states = None
next_char = tf.constant(['Good morning '])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
  result.append(next_char)

print(tf.strings.join(result)[0].numpy().decode("utf-8"))