#ASK: could make this a lot simpler, how clear vs. efficient should NEA code look (e.g. could really shorten calculate_columns and calculate_rows section!!) 
#This class is written on the notation that a matrix denoted ( 2  4 ) mathematically is represented as [[2,4],[6,8]] in python
#                                                            ( 6  8 )
#For single row matrices, formatting MUST be [[1,2,3]] NOT [1,2,3]!!! - perhaps could adapt this later using the whole type() thing

class Matrix: 
    
    def __init__(self, matrix: list): 
        self.matrix = matrix #potentially add private properties, ASK: worth extra marks? 
        self.row_number = len(matrix) 
        self.column_number = len(matrix[0])

    def __repr__(self): 
        #redefines behaviour of calling print, check what technique this is called, polymorphism? 
        result = ""
        for row in self.matrix: 
            result += " ".join(["{} ".format(x) for x in row]) #format is built in, perhaps replace code with another function/option later?
            result += "\n"
        return result
    

    def broadcast(self, other_matrix: object) -> object: #broadcasting is required to emulate the way numpy does addition. Numpy works on the basis that if you give two matrices to add together that don't have the same dimensions, it will still add them by broadcasting the second matrix if it only has one column or one row, to match the dimensions of the first matrix. 
        if self.row_number == other_matrix.row_number and self.column_number == other_matrix.column_number: 
            return other_matrix #this checks if a broadcast function is actually needed. it may be that they already have the same dimensions and thus can easily be added together. 
        if other_matrix.row_number == 1 and other_matrix.column_number == self.column_number: 
            result = [other_matrix.matrix[0] for i in range(self.row_number)] #this broadcasts the single row across all rows to make it the same dimensions as the first matrix 
            return Matrix(result) 
        if other_matrix.column_number == 1 and other_matrix.row_number == self.row_number: 
            result = [[other_matrix.matrix[n][0] for i in range(self.column_number)] for n in range(self.row_number)] #this broadcasts the single column across all columns to make it the same dimensions as the first matrix
            return Matrix(result) 
        
        raise ValueError("ERROR: Matrices dimensions are incompatible for adding, even if broadcasting is attempted.")

    def add(self, other_matrix: object) -> object: 
        other_matrix = self.broadcast(other_matrix)

        result = [[self.matrix[r][c] + other_matrix.matrix[r][c] for c in range(self.column_number)] for r in range(self.row_number)]
        #this has for loops as part of the list comprehension. the outer loop (for r in range..) iterates over each row in the matrices, the inner loop (for c in range) iterates over each column. Therefore, it acceses each element at the matching row and column number and adds them together.
        return Matrix(result) 
    
    def multiply(self, other_matrix: object) -> object: 
       
        def dot_product(row, column): 
            dot_product_result = 0
            for i in range(len(row)): #you can do len(row) or len(column) here as they must be the same LENGTH to do dot product!!!
                dot_product_result += row[i] * column[i] 
            return dot_product_result
        
        result = [[0 for c in range(other_matrix.column_number)] for r in range(self.row_number)] #as we know from theory, the dimensions of the result matrix are going to be the number of rows of the first matrix and the number of columns on the second matrix
        for i in range(self.row_number): #iterates over each row in the first matrix
            for n in range(other_matrix.column_number): #iterates over each row in the second matrix 
                result[i][n] = dot_product(self.matrix[i],[row[n] for row in other_matrix.matrix]) #matrix multiplication is essentiall the dot product of each row with each column. The row[n] for... is needed because its essentially getting a list for the column (which doesn't really exist in the current format as the lists are rows - essentially transposing the other matrix)

        return Matrix(result) 
    
    def transpose(self) -> object: 
        result = [[self.matrix[i][n] for i in range(self.row_number)] for n in range(self.column_number)]
        return Matrix(result) 

    def clip(self, lower = None, upper = None) -> object: #this does the same thing as clip in numpy. Does NOT change the NUMBER of elements, just modifies the values to the minimum and maximum values if it falls outside the range
        if upper == None: 
            upper = max(self.matrix)
        if lower == None: 
            lower = min(self.matrix)

        result = []

        for i in range(len(self.matrix)): 
            if self.matrix[i] > upper: 
                result.append(upper)
            elif self.matrix[i] < lower: 
                result.append(lower) 
            else: 
                result.append(self.matrix[i])
        
        return Matrix(result)

               





#Preliminary testing:
matrix1 = Matrix([[1,2],[3,4]])
matrix2 = Matrix([[5],[6]])
matrix3 = Matrix([[7,8],[9,1]])
print(matrix1) 
print(matrix2) 
print(matrix3) 
print(matrix1.multiply(matrix2))
print(matrix1.multiply(matrix3))
print(matrix1.add(matrix3))
print(matrix1.transpose())
matrix4 = Matrix([[0.7, 0.1, 0.2],[0.1, 0.5, 0.4],[0.02, 0.9, 0.08]])
matrix5 = Matrix([[1, 0, 0],[0, 1, 0],[0, 1, 0]])
print(matrix4.multiply(matrix5.transpose()))

    






