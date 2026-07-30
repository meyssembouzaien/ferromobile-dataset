import ctypes

lib_path = "/home/meyssem/Documents/ferromobile_project/crc-covlib/python-wrapper/crc_covlib/bin_64bit/libcrc-covlib.so"

try:
    lib = ctypes.CDLL(lib_path)
    print("✅ La bibliothèque a été chargée avec succès !")
except OSError as e:
    print("❌ Impossible de charger la bibliothèque :", e)