
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#with open('11_train_loss.txt', 'r') as file:
#    l = file.read().splitlines()
    
with open('00new3_test_loss_11.txt', 'r') as file:
    acc = file.read().splitlines()

with open('00new3_test_loss_01.txt', 'r') as file:
    acc01 = file.read().splitlines()
    
with open('00new3_test_loss_10.txt', 'r') as file:
    acc10 = file.read().splitlines()
    
   
print(acc[:2])
#l = [float(a) for a in l]
acc = [float(a) for a in acc]
acc01 = [float(a) for a in acc01]
acc10 = [float(a) for a in acc10]

x = [19+20*n for n in range(len(acc))]

'''xn = []
accn = []
acc01n = []
acc10n = []
for i in range(len(x)):
    if i%10 == 0:
        xn.append(x[i])
        accn.append(acc[i])
        acc01n.append(acc01[i])
        acc10n.append(acc10[i])'''
        
xn, accn, acc01n, acc10n = x, acc, acc01, acc10

fig, ax = plt.subplots()

#plt.plot(l)
ax.plot(xn, accn, lw = 1, label = 'тест без зануления')
ax.plot(xn, acc01n, lw = 1, label = 'тест с занулением 1-й группы каналов', linestyle='dashed')
ax.plot(xn, acc10n, lw = 1, label = 'тест с занулением 2-й группы каналов', linestyle='dotted')

ax.legend()
#ax.set_title('Точность на тесте во время обучения с занулением')
ax.set_xlabel('номер итерации')
ax.set_ylabel('ошибка')

plt.show()
plt.savefig("image.jpg")

print(acc[0])
