import matplotlib.pyplot as plt

log = '/home/kyle/school/farapy/mae/output_dir_pretrain/log.txt'

with open(log) as f:
    lines = f.readlines()

epoch_set = set()

loss_list = []
lr_list = []

lines.reverse()
for line in lines:
    epoch_index = line.find('epoch') + 8
    epoch = int(line[epoch_index:-2])
    if epoch not in epoch_set:
        lr_index = line.find('train_lr') + 11
        loss_index = line.find('train_loss') + 13
        lr = line[lr_index:loss_index-16]
        loss = line[loss_index:epoch_index-11]
        loss_list.append(float(loss))
        lr_list.append(float(lr))
        epoch_set.add(epoch)

loss_list.reverse()
lr_list.reverse()

print(loss_list)
print(lr_list)
plt.plot(loss_list)
plt.title('MAE-FaceU Pre-training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Average Error)')
plt.show()