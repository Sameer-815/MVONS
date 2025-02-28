import os, sys, json

work_dir = sys.argv[1]

for f in os.listdir(work_dir):
    if '.log.json' in f:
        records = []
        ls = open(os.path.join(work_dir, f)).readlines()
        for l in ls:
            j = json.loads(l)
            if 'mIoU' in j:
                records.append([j['mIoU'], j['iter']])
        records = sorted(records, key=lambda k: k[0])
        print(records)
        source_name = 'iter_' + str(records[-1][1]) + '.pth'
        target_name = 'best_iter_' + str(records[-1][1]) + '.pth'
        print('cp ' + os.path.join(work_dir, source_name) + ' ' + os.path.join(work_dir,  target_name))
        os.system('cp ' + os.path.join(work_dir, source_name) + ' ' + os.path.join(work_dir,  target_name))
