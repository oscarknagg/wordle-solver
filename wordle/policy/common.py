class Policy:
    def guess(self) -> str:
        raise NotImplementedError

    def update(self, output):
        raise NotImplementedError
