import matplotlib.pyplot as plt

filename = '/home/kyle/school/farapy/mae/output_dir/log.txt'

with open(filename) as f:
    content = f.read().splitlines()

losses = []
for line in content:
    idx = line.find('loss') + 7
    losses.append(float(line[idx:idx+10]))

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss (MAE)')
plt.title('MAE Training Loss')
plt.show()