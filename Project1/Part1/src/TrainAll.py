import os
import sys
import subprocess

lr = [0.001, 0.01]
neurons = [1024, 512, 128, 64]

loss = ['mse','ce']
opt  = ['sgd','adam']

total = len(lr) * len(neurons) * len(loss) * len(opt)
count = 1

try:
    os.makedirs(os.path.join('outputs','pics'))
except (FileExistsError, FileNotFoundError):
    pass
except Exception as e:
    print(f"error making folder: {e}")
    sys.exit(0)

cd = os.path.join(os.getcwd(), 'outputs')

for i in neurons:
    for j in lr:
        foldername = f"{i}_{str(j).split('.')[1]}"
        try:
            os.makedirs(os.path.join(cd, foldername))
        except (FileExistsError, FileNotFoundError):
            pass
        except Exception as e:
            print(f"error making folder {foldername}: {e}")
            sys.exit(0)
        
        for l in loss:
            for o in opt:
                subprocess.run([sys.executable, 'Train.py', '--epochs', '2000', '--lr', f'{j}', 
                                '--output', os.path.join(cd, foldername, foldername + f'_{l}_{o}'),
                                '--plot', os.path.join(cd, 'pics', foldername + f'_{l}_{o}'), 
                                '--neurons', f'{i}'] , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f'Completed {count} out of {total}')
                count += 1

