import torch , torchvision

neuralnet = torchvision.models.resnet18().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(neuralnet.parameters(), lr=0.1, momentum=0.9)

for epoch in range(100):
	for (inputs , labels) in train_loader:
		loss = loss_func(neuralnet(inputs), labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

