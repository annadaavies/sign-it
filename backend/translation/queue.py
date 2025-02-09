class QueueNode: 
    """
    A node class container for elements in the queue. 
    """
    def __init__(self, value: any): 
        """
        Initialise a queue node with given value. 
        
        Args: 
        value: The data to store in the queue node. 
        """
        self.value = value
        self.next = None
        
class Queue: 
    """
    A First-In-First-Out (FIFO) data structure implementation. 
    """
    def __init__(self): 
        """
        Initialise an empty queue. 
        """
        self.front = None
        self.rear = None
        self.count = 0
        
    def enqueue(self, value: any): 
        """
        Add an element to the end of the queue. 
        
        Args: 
        value: The value of the node to be added to the queue. 
        """
        new_node = QueueNode(value) 
        
        if self.rear is None: #Check if the queue is empty, if it is both the front and the rear pointers point to the new node.
            self.front = self.rear = new_node
        else: #If queue not empty, add the new node after the current rear and update the rear to be the new node. 
            self.rear.next = new_node
            self.rear = new_node
        
        self.count += 1 
        
    def dequeue(self) -> any: 
        """
        Remove and return ('pop') the value of the node at the front of the queue. 
        
        Returns: 
        any: The front node's value (None if the queue is empty).
        """
        if self.is_empty(): 
            return None
        
        temp_node = self.front #Store the current front node temporarily (before removed).
        self.front = temp_node.next #Move the front pointer to the next node. 
        
        if self.front is None: #If queue becomes empty after going to next node, set front and rear pointer to None. 
            self.rear = None
            
        self.count -= 1
        
        return temp_node.value
    
    def is_empty(self) -> bool: 
        """
        Check if the queue contains any nodes. 
        
        Returns: 
        bool: Flag to mark whether the queue is empty (True if queue is empty, False if queue has at least one node). 
        """
        check_empty = self.front is None
        return check_empty
    
    def peek(self) -> any: 
        """
        View the value of the node at theh front of the queue without removing it. 
        
        Returns: 
        any: The front node's value (None if the queue is empty). 
        """
        if self.front: 
            return self.front.value
        
        return None
    
    def items(self) -> list: #TODO: For function type hints, this returns a list (of any data type), can't seem to do -> list[any] or -> list?
        """
        Retrieve all elements in the queue as a list. 
        
        Returns: 
        list[any]: List of all nodes in the queue in FIFO order. 
        """
        items = []
        current_node = self.front
        
        while current_node: #Iterate from front to rear node, adding each value to items list. 
            items.append(current_node.value)
            current_node = current_node.next
            
        return items

    
    
    