import heapq

class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
        self.index = None
        self.vector = None
    
    def __lt__(self, other):
        if other is None:
            return -1
        if not isinstance(other, HeapNode):
            return -1
        return self.freq < other.freq


class HuffmanTree:
    def __init__(self):
        self.heap = []
        self.char_to_code = {}
        self.code_to_char = {}
        self.merged_nodes = None
    
    # heap tree를 우선적으로 만들어놓음
    def make_heap(self, vocab):
        for key in vocab.keys():
            node = HeapNode(key, vocab[key])
            heapq.heappush(self.heap, node)

    # heap tree안에서부터 frequency작은거부터 하나씩 뽑아서 node만들고 합침
    def merge_nodes(self):
        index = 0
        merged = None
        while len(self.heap) > 1:
            # freq 작은 heap을 2개 꺼내서 노드1, 노드2로 만듦
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            
            # 노드1, 노드2 합침
            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            # 합쳐진 node에는 index 0부터 1, 2, ... 넣는다.
            merged.index = index
            heapq.heappush(self.heap, merged)

            index += 1
        
        return merged

    def make_code_helper(self, root, current_code):
        if root is None:
            return

        if root.char is not None:
            self.char_to_code[root.char] = current_code
            self.code_to_char[current_code] = root.char
            return

        # root부터 시작해서 왼쪽, 오른쪽 노드들에 대해서 char_to_code, code_to_char채움
        self.make_code_helper(root.left, current_code + '0')
        self.make_code_helper(root.right, current_code + '1')

    def make_codes(self):
        # merge_node를 지나고나면 heap에는 1개만 남음 : root 
        root = heapq.heappop(self.heap)
        current_code = ''
        self.make_code_helper(root, current_code)
        
    def build(self, vocab):
        self.make_heap(vocab)
        merged = self.merge_nodes()
        self.make_codes()

        return self.char_to_code, merged


#code_sign 은 root로부터 leaf까지 왼쪽이면 -1, 오른쪽이면 1 적힌 리스트
def code_to_id(char_to_code, root, vocab):
    node = root
    idx = []
    code_sign = []
    for word in vocab:
        if word == '\s':
            continue
        else:
            temp0 = []
            temp1 = []
            codes = char_to_code[word]
        # print('codes: ', codes, '// word_id: ', word_id)
        for code in codes:
            if code == '0':
                temp0.append(node.index)
                temp1.append(-1)
                node = node.left
                # print('code 0 is run')
            elif code == '1':
                temp0.append(node.index)
                temp1.append(1)
                node = node.right
                # print('code 1 is run')
            if node.index is None:
                node = root
                # print('PASSED')
                break
        idx.append(temp0)
        code_sign.append(temp1)
    return idx, code_sign


def test():
    vocab = {"a": 4, 'b': 6, 'c': 3, 'd': 2, 'e': 2}
    huffman = HuffmanTree()
    char_to_code, root = huffman.build(vocab)
    print("char_to_code: ", char_to_code)
    code_idx, code_sign = code_to_id(char_to_code, root, vocab)
    print(code_idx[0])
    print(code_sign[0])

# test()