import json , pdb

def process(infile , outfile):
    fr = open(infile , 'r',encoding='utf-8')
    lines = fr.read().splitlines()

    iterm_dict = {}
    for ndx , term in enumerate(lines):
        iterm_dict[term] = ndx
    vocab_json = json.JSONEncoder().encode(iterm_dict)
    fr.close()
    fw = open(outfile , 'w' , encoding = 'utf-8')
    fw.write(vocab_json)
    fw.flush()
    fw.close()
if __name__ == '__main__':
    infile = '../data/text.data/vocab_processed.txt'
    outfile = '../data/text.data/vocab_processed.json'
    process(infile , outfile)
    encoder = json.load(open(outfile, encoding="utf-8"))
    print(encoder)