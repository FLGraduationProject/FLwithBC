import torch

def test(clientID, client_model, testLoader, device):

    # This is for test data
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(testLoader):
            image, label = data

            image = image.to(device)
            label = label.to(device)
            
            output = client_model(image)

            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if (batch_idx + 1) == len(testLoader):
                print(clientID, "Test_Acc:{:.3f}%".format(100. * correct / total))

    return 100. * correct / total