import torch


class DiagonalNeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        num_layers=5,
        num_features=10,
        share_parameters=False,
        device="cuda",
        init_method=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.share_parameters = share_parameters
        self.init_method = init_method
        blocks = list()
        if share_parameters:
            block = torch.ones(num_features, requires_grad=True, device=device)
            block = torch.nn.Parameter(block)
            for i in range(num_layers):
                blocks.append(block)
        else:
            for i in range(num_layers):
                block = torch.randn(num_features, requires_grad=True, device=device)
                block = torch.nn.Parameter(block)
                blocks.append(block)
        self.blocks = torch.nn.ParameterList(blocks)
        self.initialize_parameters()

    @torch.no_grad()
    def initialize_parameters(self):
        if self.init_method is None:
            for block in self.blocks:
                torch.nn.init.normal_(block, mean=0.0, std=1.0)
        elif self.init_method == "uniform":
            for block in self.blocks:
                torch.nn.init.uniform_(block, a=-1.0, b=1.0)
                # Initialize parameters with a uniform distribution between -1 and 1
        elif self.init_method == "normal":
            for block in self.blocks:
                torch.nn.init.normal_(block, mean=0.0, std=1.0)
                # Initialize parameters with a normal distribution with mean 0 and std 1
        elif self.init_method == "xavier":
            for block in self.blocks:
                torch.nn.init.xavier_normal_(block, gain=1.0)
                # Initialize parameters with Xavier normal initialization

    def forward(self, x: torch.Tensor):
        # x.shape=(batch_size, num_features)
        for i in range(self.num_layers):
            block = self.blocks[i]
            x = block * x

        return x


class DiagonalNeuralNetworkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples=1000,
        num_features=10,
        noise_scaling_factor=1.0,
        num_layers=5,
        share_parameters=False,
        init_method="uniform",
        device="cuda",
    ):
        self.num_samples = num_samples
        self.num_features = num_features
        self.noise_scaling_factor = noise_scaling_factor
        self.device = device

        # Create a true model to generate data
        self.true_model = DiagonalNeuralNetwork(
            num_layers=num_layers,
            num_features=num_features,
            share_parameters=share_parameters,
            device=device,
            init_method=init_method,
        )

        # Generate input data
        self.x = torch.rand(num_samples, num_features, device=device)

        # Generate output data using the true model
        with torch.no_grad():
            self.y = self.true_model(self.x)

        # Add noise to the output
        noise = torch.randn_like(self.y) * noise_scaling_factor
        self.y += noise

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = DiagonalNeuralNetworkDataset(
        num_samples=1000, num_features=10, num_layers=2, noise_scaling_factor=0.1
    )
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Print some information
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"True model structure: {dataset.true_model}")

    # Get a sample
    x_sample, y_sample = dataset[0]
    print(f"Sample input shape: {x_sample.shape}")
    print(f"Sample output shape: {y_sample.shape}")

    # Iterate through the dataloader
    for batch_x, batch_y in dataloader:
        print(f"Batch input shape: {batch_x.shape}")
        print(f"Batch output shape: {batch_y.shape}")
        break  # Just print the first batch

    train_size = int(0.8 * len(dataset))  # 80% training
    test_size = len(dataset) - train_size  # 20% testing
    from torch.utils.data import random_split

    print(len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(len(train_dataset), len(test_dataset))
