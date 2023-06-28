import pickle
import matplotlib.pyplot as plt

with open('./test.pkl', 'rb') as file:
    noadv = pickle.load(file)

with open('./testadv2.pkl', 'rb') as file:
    adv = pickle.load(file)

# plt.title('before update')
plt.title('after update')
plt.plot(adv[1], label='per = 0.2')
plt.plot(noadv[1], label='no per')
plt.legend()
plt.show()

