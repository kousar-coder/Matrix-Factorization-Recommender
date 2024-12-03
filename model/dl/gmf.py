import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

class MFModel(nn.Module):
    """The Matrix Factorization model."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        """Initializes model parameters."""
        super().__init__()
        self.user_embeddings = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embeddings = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # Uniform initialization
        self.user_embeddings.weight.data.uniform_(0.5, 1.0)
        self.item_embeddings.weight.data.uniform_(0.5, 1.0)

        self.affine_tranform = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""
        user_embeddings = self.user_embeddings(users)
        item_embeddings = self.item_embeddings(items)
        out = self.affine_tranform(user_embeddings * item_embeddings)
        return out

# Data loading and preprocessing
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("ml-latest-small/ratings.csv")
    df.drop("timestamp", axis=1, inplace=True)

    # Normalize ratings
    rating, min_rating, max_rating = df["rating"], df["rating"].min(), df["rating"].max()
    df["rating"] = (rating - min_rating) / (max_rating - min_rating)
    print(f"rating is from {df['rating'].min()} to {df['rating'].max()}")

    # Reassign ratings for recommendation threshold
    cond = df["rating"] < 0.5
    df["rating"] = df["rating"].where(cond, 0)
    df["rating"] = df["rating"].where(~cond, 1)

    # Encoding movieId and userId
    enc_movie = {movie_id: idx for idx, movie_id in enumerate(df["movieId"].unique())}
    df["movieId"] = [enc_movie[movie_id] for movie_id in df["movieId"]]
    print(f"movieId is from {df['movieId'].min()} to {df['movieId'].max()}")

    enc_user = {user_id: idx for idx, user_id in enumerate(df["userId"].unique())}
    df["userId"] = [enc_user[user_id] for user_id in df["userId"]]
    print(f"userId is from {df['userId'].min()} to {df['userId'].max()}")

    # PyTorch Dataset
    class MovieLensSmall(torch.utils.data.Dataset):
        def __init__(self, df: pd.DataFrame) -> None:
            self.df = df

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]
            return torch.tensor(row['userId']), torch.tensor(row['movieId']), torch.tensor(row['rating'])

    train_dataloader = torch.utils.data.DataLoader(
        MovieLensSmall(df),
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Adjust based on your system capabilities
    )

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MFModel(num_users=len(enc_user), num_items=len(enc_movie), embedding_dim=10).to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Training loop
    log_idx = 1_000
    for epoch in range(10):
        running_loss = 0.0
        for idx, (users, items, ratings) in enumerate(train_dataloader):
            users = users.to(device).long()
            items = items.to(device).long()
            ratings = ratings.to(device)

            optimizer.zero_grad()

            outputs = model(users, items).reshape(-1)

            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % log_idx == log_idx - 1:
                print(f"Epoch {epoch} | Steps: {idx + 1:<4} | Loss: {running_loss / log_idx:.3f}")
                running_loss = 0.0

        # Optionally save the model after every epoch
        torch.save(model.state_dict(), f"mf_epoch_{epoch}.pth")
