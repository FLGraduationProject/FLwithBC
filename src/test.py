import torch

def test(client):
    testloader = client.testloader

    # This is for test data
    testmodel = client.model
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            image, label = data

            image = image.cuda()
            label = label.cuda()
            
            output = testmodel(image)

            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if (batch_idx + 1) == len(testloader):
                print(client.clientID, "Test_Acc:{:.3f}%".format(100. * correct / total))
    return 100. * correct / total