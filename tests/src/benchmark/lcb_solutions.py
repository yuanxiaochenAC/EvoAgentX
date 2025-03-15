
codegen_solution = """
```python
def can_transform_to_abc(s):
    target = "abc"
    
    # If already in correct order, no swap needed
    if s == target:
        return "YES"
    
    # Check if exactly one swap can convert s to "abc"
    differences = [(s[i], target[i]) for i in range(3) if s[i] != target[i]]
    
    # A single swap means exactly two characters must be out of place
    if len(differences) == 2 and differences[0] == differences[1][::-1]:
        return "YES"
    
    return "NO"

def process_test_cases():
    t = int(input().strip())
    results = []
    for _ in range(t):
        s = input().strip()
        results.append(can_transform_to_abc(s))
    print("\\n".join(results))

# Run the function to process input
process_test_cases()
```
"""

codegen_solution2 = """

def solve_case():
    n = int(input())
    digits = list(map(int, input().split()))
    
    max_product = 0
    
    # Try adding 1 to each position
    for i in range(n):
        # Calculate product with current digit increased by 1
        product = 1
        for j in range(n):
            if j == i:
                product *= (digits[j] + 1)
            else:
                product *= digits[j]
        max_product = max(max_product, product)
    
    return max_product

# Read number of test cases
t = int(input())

# Process each test case
for _ in range(t):
    print(solve_case())

"""


codegen_solution3 = """
The following are the solutions for the problem:

```python
def solve_case():
    n, k = map(int, input().split())
    s = input()
    
    # If no black cells, return 0
    if 'B' not in s:
        return 0
    
    operations = 0
    i = 0
    
    while i < n:
        # Find next black cell
        if s[i] == 'B':
            # Cover k cells starting from this black cell
            operations += 1
            i += k
        else:
            i += 1
            
    return operations

# Read number of test cases
t = int(input())

# Process each test cases
for _ in range(t):
    print(solve_case())
```
"""