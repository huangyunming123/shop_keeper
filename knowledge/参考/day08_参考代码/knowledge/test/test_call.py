from knowledge.processor.import_process.state import ImportGraphState


class Node1:
    """
    节点1配置项
    """
    def __call__(self, state: ImportGraphState):
        print("节点1 call")

    def process(self):
        print("节点1 process")


if __name__ == "__main__":
    node1 = Node1()
    # node1.process()

    graph.invoke(state)

    node1(state)