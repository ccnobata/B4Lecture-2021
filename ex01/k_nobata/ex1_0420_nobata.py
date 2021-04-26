import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

data, samplerate = sf.read("C:/Users/nobat/b4python/新規録音2.wav")  # 音声ファイル読込

t = np.arange(0, len(data)) / samplerate  # 横軸の設定

F_size = 1024  # フレーム幅
OVERLAP = F_size / 2  # 半分オーバーラップ
F_num = data.shape[0]  # フレームの要素数
Ts = float(F_num) / samplerate  # 波形の長さ
S_num = int(F_num // (F_size - OVERLAP) - 1)  # 短時間区間数

window = np.hamming(F_size)  # ハミング窓作成
spec = np.zeros([S_num, F_size], dtype=np.complex)  # 収納用配列
pos = 0  # 初期化

for fft_index in range(S_num):
    frame = data[int(pos) : int(pos + F_size)]  # フレーム幅の短時間区間を取り出す
    if len(frame) == F_size:
        windowed = window * frame  # 窓かけ
        fft_result = np.fft.fft(windowed)  # フーリエ変換
        for i in range(len(spec[fft_index])):
            spec[fft_index][i] = fft_result[i]  # 保存

        pos += F_size - OVERLAP  # 次の区間へシフト

ifft_result = np.zeros(F_num)  # 収納用配列
pos = 0  # 初期化
for ifft_index in range(S_num - 1):
    ifft_data = np.fft.ifft(spec[ifft_index])  # 逆フーリエ変換
    ifft_result[int(pos) : int(pos + F_size)] += np.real(ifft_data)  # 保存
    pos += F_size - OVERLAP  # 次の区間へシフト

fig = plt.figure()
# 元の音声データ
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(t, data)
plt.title("Original signal")
plt.ylabel("Magnitude")

# スペクトログラム
ax2 = fig.add_subplot(3, 1, 2)
ax2.imshow(
    np.log(abs(spec[:, :512].T)),
    extent=[0, Ts, 0, samplerate / 2],
    aspect="auto",
)

plt.title("Spectrogram")
plt.ylabel("Frequency[Hz]")

# 逆フーリエ展開後
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(t, ifft_result)
plt.title("Re-synthesized signal")
plt.xlabel("Times[s]")
plt.ylabel("Magnitude")

# 画像表示と保存
plt.tight_layout()
plt.show()
plt.close()