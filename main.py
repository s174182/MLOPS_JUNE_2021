import sys
import argparse
import matplotlib.pyplot as plt
import torch

from data import mnist
from model import MyAwesomeModel

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        epochs=5
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = MyAwesomeModel()
        train_loader, testloader = mnist()
        steps = 0
        running_loss = 0
        fig, ax = plt.subplots()
        for e in range(epochs):
            model.train()
            for images, labels in trainloader:
                steps += 1
                
                # Flatten images into a 784 long vector
                images.resize_(images.size()[0], 784)
                
                optimizer.zero_grad()
                
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
        
        plt.title('Training loss')
        ax.scatter(e,running_loss,color="red")
        plt.show()
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = torch.load(args.load_model_from)
        _, test_set = mnist()

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    