import os

os.system('pip install -r requirements.txt')
if not os.path.exists('logs'):
    os.mkdir('logs')