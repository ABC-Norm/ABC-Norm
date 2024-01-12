import torch
import time

from tqdm import tqdm

def accum_acc(logits, label):
    value, index = torch.max(logits, dim=1)
    acc = torch.sum(index.data == label.data).type(torch.float)
    return acc

def trainIter(epoch, epochs, model, trainloader, loss_funcs, optimizer, device="cuda", weight=None, **kwargs):
    model.train()

    losses = 0
    ce_loss = 0
    total = 0
    acc = 0

    pbar = tqdm(trainloader)
    for idx, (image, label, path) in enumerate(pbar):
        batch_size = image.size(0)
        image, label = image.to(device), label.to(device)

        logits = model(image)

        loss, ce = loss_funcs(logits, label, weight=weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += batch_size
        losses += loss.detach() * batch_size
        if ce:
            ce_loss += ce.detach() * batch_size

        acc += accum_acc(logits, label)
        pbar.set_description("Training, Epochs: {}/{}, Loss: {:.4f}, CELoss: {:.4f}, Acc: {:.2f}%"
                             .format(epoch, epochs, losses/total, ce_loss/total, 100.0*acc/total))
    avg_loss = losses / total
    avg_ce_loss = ce_loss / total
    avg_acc = acc / total
    return avg_loss, avg_ce_loss, avg_acc

def validate(model, validloader, device="cuda", **kwargs):
    model.eval()

    total = 0
    acc = 0

    pbar = tqdm(validloader)
    with torch.no_grad():
        for idx, (image, label, path) in enumerate(pbar):
            batch_size = image.size(0)
            image, label = image.to(device), label.to(device)

            logits = model(image)

            total += batch_size
            acc += accum_acc(logits, label)

            pbar.set_description("Validation, Acc: {:.2f}%"
                                 .format(100.0*acc/total))
    avg_acc = acc / total
    return avg_acc

def validate_with_fcr(model, validloader, count, device="cuda"):
    model.eval()

    total = 0
    acc = 0

    f_acc = 0
    f_count = 0
    c_acc = 0
    c_count = 0
    r_acc = 0
    r_count = 0

    entropy = 0

    count = torch.tensor(count, dtype=torch.long, device=device)
    
    pbar = tqdm(validloader)
    with torch.no_grad():
        for i, (imgs, targets, path) in enumerate(pbar):
            batch_size = imgs.size(0)
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)

            pred = torch.softmax(logits, dim=1)
            entropy += -torch.sum(pred * torch.log_softmax(logits, dim=1)).detach()

            value, index = torch.max(logits, dim=1)

            for idx, row in enumerate(targets):
                c = count[row]
                if c > 100:
                    f_count += 1
                    f_acc += (index.data[idx] == targets.data[idx]).type(torch.float)
                elif c > 20:
                    c_count += 1
                    c_acc += (index.data[idx] == targets.data[idx]).type(torch.float)
                else:
                    r_count += 1
                    r_acc += (index.data[idx] == targets.data[idx]).type(torch.float)
                
            acc += torch.sum(index.data == targets.data).type(torch.float)
            total += batch_size

            acc_per = 100 * acc / total
            f_acc_per = 100 * f_acc / f_count
            c_acc_per = 100 * c_acc / c_count
            r_acc_per = 100 * r_acc / r_count

            pbar.set_description("Validation, Total Acc: {:.2f}%, Many: {:.2f}%, Median: {:.2f}%, Low: {:.2f}, Entropy: {:.2f}".format(acc_per, f_acc_per, c_acc_per, r_acc_per, entropy/total))

    avg_acc = acc / total
    return avg_acc
