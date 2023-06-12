from __future__ import print_function,division
import sys
import numpy as np
import torch
from prose.alphabets import Uniprot21
from Bio.Seq import Seq
from Bio import SeqIO
from multiprocessing import Process
import os
import time
from prose.models.multitask import ProSEMT

# ********* SETTINGS **********



FASTA_FILE_PATH = "../BioEmbedding/dataset/globins/globins.fasta"
OUT_DIR = "../BioEmbedding/dataset/globins/embeddings/prose"
MAX_CHUNK_SIZE = 1024
FAST_MODE = False


POOL = "none"
ALL_STACK = False

# ******************************



def predict(id, chunk_index, query_sequence, write_to_path=False, pool='none'):
    
    start = time.time()

    query_sequence = bytes(query_sequence, "utf-8")

    model = ProSEMT.load_pretrained()
    model.eval()
  
    
    if len(query_sequence) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1,n), dtype=np.float32)
        return z

    alphabet = Uniprot21()
    query_sequence = query_sequence.upper()
    # convert to alphabet
    query_sequence = alphabet.encode(query_sequence)
    query_sequence = torch.from_numpy(query_sequence)
    

    # embed the sequence
    with torch.no_grad():
        query_sequence = query_sequence.long().unsqueeze(0)
        
        z = None
        
        if ALL_STACK:
            z = model.transform(query_sequence) # all the network stack
        else:
            z = model(query_sequence) # only the last z layer
       
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()

    if write_to_path != False:
        np.save(write_to_path, z)

    end = time.time()

    print(f"Time for embedding: {end - start}", file=sys.stderr, flush=True)

    return z



def main():

    pid = os.getpid()
    print(f'{pid}, {FASTA_FILE_PATH}', file=sys.stderr, flush=True)

    # check if the output directory exists
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)        

    for count, seqrecord in enumerate(SeqIO.parse(FASTA_FILE_PATH, "fasta")):

        seq_id = seqrecord.id

        # check if the file already exists
        if os.path.exists(os.path.join(OUT_DIR, f"{seq_id}.npy")):
            print(f"Skipping {seq_id} because already exists", file=sys.stderr, flush=True)
            continue

        seq_string = str(seqrecord.seq)
        seq_string = seq_string.replace(" ", "").replace("\n", "")

        if set(seq_string).issubset(set(["A", "C", "G", "T"])):
            seq_string = str(Seq(seq_string).translate(stop_symbol=""))
            print("The nucleotides sequence for ", seq_id, " has been translated", file=sys.stderr, flush=True)
        
        # split the sequence in chunks such that each chunk has approximately the same length
        N = int(np.ceil(len(seq_string) / MAX_CHUNK_SIZE)) # number of chunks
        chunks = [seq_string[(i*len(seq_string))//N:((i+1)*len(seq_string))//N] for i in range(N)] # list of chunks

        lens = [len(chunk) for chunk in chunks]

        if (FAST_MODE):
            
            sequence_embedding = []
            for chunk_index, chunk in enumerate(chunks):
                print(f"Predicting the embedding {count+1}, chunk {chunk_index+1}/{len(chunks)}", file=sys.stderr, flush=True)
                z = predict(id=seq_id, chunk_index=chunk_index, query_sequence=chunk, write_to_path=False)    
                sequence_embedding.append(z)

            # can happen that che subsequences are not of the same length, in this case pad them with the mean value
            max_len = max([len(z) for z in sequence_embedding]) # z is a np array size (chunk_size x 1280)
            for i, z in enumerate(sequence_embedding):
                if len(z) < max_len:
                    sequence_embedding[i] = np.append(z, [np.mean(z, axis=0)], axis=0) # it is enough to append only one value, since the max difference between chunks is 1

            sequence_embedding = np.array(sequence_embedding)
            
            # save the embedding
            np.save(os.path.join(OUT_DIR, f"{seq_id}.npy"), sequence_embedding)
        
        else:
            
            # save each chunk in a separate file
            for chunk_index, chunk in enumerate(chunks):
                print(f"Predicting the embedding {count+1}, chunk {chunk_index+1}/{len(chunks)}", file=sys.stderr, flush=True)
                # check if the file already exists
                if os.path.exists(os.path.join(OUT_DIR, f"{seq_id}_chunk:{chunk_index}.npy")):
                    print(f"Skipping {seq_id}_chunk:{chunk_index+1} because already exists", file=sys.stderr, flush=True)
                    continue
                # run predict in a separate process
                p = Process(target=predict, args=(seq_id, chunk_index, chunk, os.path.join(OUT_DIR, f"{seq_id}_chunk:{chunk_index}.npy")))
                p.start()
                p.join()
            
            # load the chunks and recombine them
            print(f"Recombining the chunks for {seq_id}", file=sys.stderr, flush=True)
            sequence_embedding = []
            for chunk_index in range(len(chunks)):
                z = np.load(os.path.join(OUT_DIR, f"{seq_id}_chunk:{chunk_index}.npy"))
                sequence_embedding.append(z)
                # remove the chunk file
                os.remove(os.path.join(OUT_DIR, f"{seq_id}_chunk:{chunk_index}.npy"))

            # perform the padding as before
            max_len = max([len(z) for z in sequence_embedding]) # z is a np array size (chunk_size x 1280)
            for i, z in enumerate(sequence_embedding):
                if len(z) < max_len:
                    sequence_embedding[i] = np.append(z, [np.mean(z, axis=0)], axis=0) # it is enough to append only one value, since the max difference between chunks is 1

            sequence_embedding = np.array(sequence_embedding)
            
            # save the embedding
            np.save(os.path.join(OUT_DIR, f"{seq_id}.npy"), sequence_embedding)


    print(f'{pid}, DONE', file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
