import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Determine how often to print the batch loss while training/validating. 
# We set this at `100` to avoid clogging the notebook.
PRINT_EVERY = 100



def train(train_loader, encoder, decoder, criterion, optimizer, vocab_size,
          epoch, total_step, start_step=1, start_loss=0.0):
    """Train the model for one epoch using the provided parameters. Save 
    checkpoints every 100 steps. Return the epoch's average train loss."""

    # Switch to train mode
    encoder.train()
    decoder.train()

    # Keep track of train loss
    total_loss = start_loss

    # Start time for every 100 steps
    start_train_time = time.time()

    for i_step in range(start_step, total_step + 1):
        # Randomly sample a caption length, and sample indices with that length
        indices = train_loader.dataset.get_indices()
        # Create a batch sampler to retrieve a batch with the sampled indices
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch
        for batch in train_loader:
            images, captions = batch[0], batch[1]
            break 
        # Move to GPU if CUDA is available
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        # Pass the inputs through the CNN-RNN model
        features = encoder(images)
        outputs = decoder(features, captions)

        # Calculate the batch loss
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        # Zero the gradients. Since the backward() function accumulates 
        # gradients, and we don’t want to mix up gradients between minibatches,
        # we have to zero them out at the start of a new minibatch
        optimizer.zero_grad()
        # Backward pass to calculate the weight gradients
        loss.backward()
        # Update the parameters in the optimizer
        optimizer.step()

        total_loss += loss.item()

        # Get training statistics
        stats = "Epoch %d, Train step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f" \
                % (epoch, i_step, total_step, time.time() - start_train_time,
                   loss.item(), np.exp(loss.item()))
        # Print training statistics (on same line)
        print("\r" + stats, end="")
        sys.stdout.flush()

        # Print training stats (on different line), reset time and save checkpoint
        if i_step % PRINT_EVERY == 0:
            print("\r" + stats)
            filename = os.path.join("state_ntrang", "train-model-{}{}.pkl".format(epoch, i_step))
            save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch, i_step)
            start_train_time = time.time()
            
    return total_loss / total_step

def save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch, train_step=1):
    """Save the following to filename at checkpoints: encoder, decoder,
    optimizer, total_loss, epoch, and train_step."""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "total_loss": total_loss,
                "epoch": epoch,
                "train_step": train_step,
               }, filename)

def save_val_checkpoint(filename, encoder, decoder, total_loss,
    total_bleu_4, epoch, val_step=1):
    """Save the following to filename at checkpoints: encoder, decoder,
    total_loss, total_bleu_4, epoch, and val_step"""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "total_loss": total_loss,
                "total_bleu_4": total_bleu_4,
                "epoch": epoch,
                "val_step": val_step,
               }, filename)

def save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
               val_bleu, val_bleus, epoch):
    """Save at the end of an epoch. Save the model's weights along with the 
    entire history of train and validation losses and validation bleus up to 
    now, and the best Bleu-4."""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_bleu": val_bleu,
                "val_bleus": val_bleus,
                "epoch": epoch
               }, filename)

def early_stopping(val_bleus, patience=3):
    """Check if the validation Bleu-4 scores no longer improve for 3 
    (or a specified number of) consecutive epochs."""
    # The number of epochs should be at least patience before checking
    # for convergence
    if patience > len(val_bleus):
        return False
    latest_bleus = val_bleus[-patience:]
    # If all the latest Bleu scores are the same, return True
    if len(set(latest_bleus)) == 1:
        return True
    max_bleu = max(val_bleus)
    if max_bleu in latest_bleus:
        # If one of recent Bleu scores improves, not yet converged
        if max_bleu not in val_bleus[:len(val_bleus) - patience]:
            return False
        else:
            return True
    # If none of recent Bleu scores is greater than max_bleu, it has converged
    return True

def word_list(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding words as a list.
    """
    word_list = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            word_list.append(word)
    return word_list

def clean_sentence(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding sentence (as a single Python string).
    """
    sentence = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            sentence.append(word)
    sentence = " ".join(sentence)
    return sentence

def get_prediction(data_loader, encoder, decoder, vocab):
    """Loop over images in a dataset and print model's top three predicted 
    captions using beam search."""
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title("Sample Image")
    plt.show()
    if torch.cuda.is_available():
        image = image.cuda()
    features = encoder(image).unsqueeze(1)
    print ("Caption without beam search:")
    output = decoder.sample(features)
    sentence = clean_sentence(output, vocab)
    print (sentence)

    print ("Top captions using beam search:")
    outputs = decoder.sample_beam_search(features)
    # Print maximum the top 3 predictions
    num_sents = min(len(outputs), 3)
    for output in outputs[:num_sents]:
        sentence = clean_sentence(output, vocab)
        print (sentence)