#!/usr/bin/env python3

"""
Database Tools Examples for EvoAgentX

This file demonstrates how to use various database toolkits:
- MongoDBToolkit: Document database operations
- PostgreSQLToolkit: Relational database operations  
- FaissToolkit: Vector database for semantic search

Each toolkit provides comprehensive database management capabilities with automatic
storage management and support for complex queries.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path to import from evoagentx
sys.path.append(str(Path(__file__).parent.parent))

from evoagentx.tools import (
    MongoDBToolkit,
    PostgreSQLToolkit
)
from evoagentx.tools.database_faiss import FaissToolkit


def run_mongodb_examples():
    """Run examples using MongoDBToolkit for document database operations."""
    print("\n===== MONGODB TOOLKIT EXAMPLES =====\n")
    
    try:
        # Initialize MongoDB toolkit with default storage
        toolkit = MongoDBToolkit(
            name="DemoMongoDBToolkit",
            database_name="demo_db",
            auto_save=True
        )
        
        print("✓ MongoDBToolkit initialized with default storage")
        
        # Get available tools
        execute_tool = toolkit.get_tool("mongodb_execute_query")
        find_tool = toolkit.get_tool("mongodb_find")
        update_tool = toolkit.get_tool("mongodb_update")
        delete_tool = toolkit.get_tool("mongodb_delete")
        info_tool = toolkit.get_tool("mongodb_info")
        
        print(f"✓ Available tools: {[tool.name for tool in toolkit.get_tools()]}")
        
        # Example 1: Insert product data
        print("\n1. Inserting product data...")
        products = [
            {"id": "P001", "name": "Laptop", "category": "Electronics", "price": 999.99, "stock": 50, "brand": "TechCorp"},
            {"id": "P002", "name": "Wireless Mouse", "category": "Electronics", "price": 29.99, "stock": 100, "brand": "TechCorp"},
            {"id": "P003", "name": "Desk Chair", "category": "Furniture", "price": 199.99, "stock": 25, "brand": "ComfortCo"},
            {"id": "P004", "name": "Coffee Table", "category": "Furniture", "price": 149.99, "stock": 15, "brand": "ComfortCo"},
            {"id": "P005", "name": "Smartphone", "category": "Electronics", "price": 799.99, "stock": 75, "brand": "MobileTech"}
        ]
        
        insert_result = execute_tool(
            query=json.dumps(products),
            query_type="insert",
            collection_name="products"
        )
        
        if insert_result.get("success"):
            print(f"✓ Successfully inserted {len(products)} products")
            print(f"  Documents inserted: {insert_result.get('data', {}).get('inserted_count', 'Unknown')}")
        else:
            print(f"❌ Insert failed: {insert_result.get('error', 'Unknown error')}")
            return
        
        # Example 2: Find electronics products
        print("\n2. Finding electronics products...")
        find_result = find_tool(
            collection_name="products",
            filter='{"category": "Electronics"}',
            sort='{"price": -1}',
            limit=5
        )
        
        if find_result.get("success"):
            electronics = find_result.get("data", [])
            print(f"✓ Found {len(electronics)} electronics products:")
            for product in electronics:
                name = product.get('name', 'Unknown')
                price = product.get('price', 0)
                brand = product.get('brand', 'Unknown')
                print(f"  - {name}: ${price} ({brand})")
        else:
            print(f"❌ Find failed: {find_result.get('error', 'Unknown error')}")
        
        # Example 3: Update product prices
        print("\n3. Updating product prices (10% discount on electronics)...")
        update_result = update_tool(
            collection_name="products",
            filter='{"category": "Electronics"}',
            update='{"$mul": {"price": 0.9}}',
            multi=True
        )
        
        if update_result.get("success"):
            updated_count = update_result.get("data", {}).get("modified_count", 0)
            print(f"✓ Updated {updated_count} electronics products with 10% discount")
        else:
            print(f"❌ Update failed: {update_result.get('error', 'Unknown error')}")
        
        # Example 4: Complex aggregation query
        print("\n4. Running aggregation query (average price by category)...")
        aggregation_pipeline = [
            {"$group": {"_id": "$category", "avg_price": {"$avg": "$price"}, "total_stock": {"$sum": "$stock"}}},
            {"$sort": {"avg_price": -1}}
        ]
        
        agg_result = execute_tool(
            query=json.dumps(aggregation_pipeline),
            query_type="aggregate",
            collection_name="products"
        )
        
        if agg_result.get("success"):
            categories = agg_result.get("data", [])
            print(f"✓ Category analysis:")
            for category in categories:
                cat_name = category.get('_id', 'Unknown')
                avg_price = category.get('avg_price', 0)
                total_stock = category.get('total_stock', 0)
                print(f"  - {cat_name}: Avg price ${avg_price:.2f}, Total stock: {total_stock}")
        else:
            print(f"❌ Aggregation failed: {agg_result.get('error', 'Unknown error')}")
        
        # Example 5: Delete furniture products
        print("\n5. Testing delete functionality...")
        delete_result = delete_tool(
            collection_name="products",
            filter='{"category": "Furniture"}',
            multi=True
        )
        
        if delete_result.get("success"):
            deleted_count = delete_result.get("data", {}).get("deleted_count", 0)
            print(f"✓ Deleted {deleted_count} furniture products")
        else:
            print(f"❌ Delete failed: {delete_result.get('error', 'Unknown error')}")
        
        # Example 6: Get database information
        print("\n6. Getting database information...")
        info_result = info_tool()
        
        if info_result.get("success"):
            info = info_result.get("data", {})
            print(f"✓ Database info:")
            print(f"  - Database: {info.get('database_name', 'Unknown')}")
            
            # Safely handle collections display
            collections = info.get('collections', [])
            if isinstance(collections, (list, tuple)) and collections:
                print(f"  - Collections: {', '.join(collections)}")
            elif collections:
                print(f"  - Collections: {collections}")
            else:
                print("  - Collections: None")
            
            print(f"  - Total documents: {info.get('total_documents', 'Unknown')}")
        else:
            print(f"❌ Info failed: {info_result.get('error', 'Unknown error')}")
        
        print("\n✓ MongoDB examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running MongoDB examples: {str(e)}")


def run_postgresql_examples():
    """Run examples using PostgreSQLToolkit for relational database operations."""
    print("\n===== POSTGRESQL TOOLKIT EXAMPLES =====\n")
    
    try:
        # Initialize PostgreSQL toolkit with default storage
        toolkit = PostgreSQLToolkit(
            name="DemoPostgreSQLToolkit",
            database_name="demo_db",
            auto_save=True
        )
        
        print("✓ PostgreSQLToolkit initialized with default storage")
        
        # Get available tools
        execute_tool = toolkit.get_tool("postgresql_execute")
        find_tool = toolkit.get_tool("postgresql_find")
        update_tool = toolkit.get_tool("postgresql_update")
        create_tool = toolkit.get_tool("postgresql_create")
        delete_tool = toolkit.get_tool("postgresql_delete")
        info_tool = toolkit.get_tool("postgresql_info")
        
        print(f"✓ Available tools: {[tool.name for tool in toolkit.get_tools()]}")
        
        # Example 1: Create tables
        print("\n1. Creating database tables...")
        
        # Create employees table
        create_employees_sql = """
        CREATE TABLE IF NOT EXISTS employees (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            department VARCHAR(50) NOT NULL,
            salary DECIMAL(10,2) NOT NULL,
            hire_date DATE NOT NULL,
            is_active BOOLEAN DEFAULT TRUE
        );
        """
        
        create_result = create_tool(create_employees_sql)
        if create_result.get("success"):
            print("✓ Created employees table")
        else:
            print(f"❌ Create table failed: {create_result.get('error', 'Unknown error')}")
            return
        
        # Create projects table
        create_projects_sql = """
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            start_date DATE NOT NULL,
            end_date DATE,
            budget DECIMAL(12,2),
            status VARCHAR(20) DEFAULT 'active'
        );
        """
        
        create_result = create_tool(create_projects_sql)
        if create_result.get("success"):
            print("✓ Created projects table")
        else:
            print(f"❌ Create table failed: {create_result.get('error', 'Unknown error')}")
        
        # Example 2: Insert employee data
        print("\n2. Inserting employee data...")
        insert_employees_sql = """
        INSERT INTO employees (name, email, department, salary, hire_date) VALUES
        ('Alice Johnson', 'alice.johnson@company.com', 'Engineering', 85000.00, '2023-01-15'),
        ('Bob Smith', 'bob.smith@company.com', 'Marketing', 72000.00, '2023-02-01'),
        ('Carol Davis', 'carol.davis@company.com', 'Engineering', 90000.00, '2022-11-10'),
        ('David Wilson', 'david.wilson@company.com', 'Sales', 68000.00, '2023-03-20'),
        ('Eva Brown', 'eva.brown@company.com', 'HR', 65000.00, '2023-01-08')
        ON CONFLICT (email) DO NOTHING;
        """
        
        insert_result = execute_tool(insert_employees_sql)
        if insert_result.get("success"):
            print("✓ Successfully inserted employee data")
        else:
            print(f"❌ Insert failed: {insert_result.get('error', 'Unknown error')}")
        
        # Example 3: Insert project data
        print("\n3. Inserting project data...")
        insert_projects_sql = """
        INSERT INTO projects (name, description, start_date, end_date, budget, status) VALUES
        ('Website Redesign', 'Modernize company website with new design and features', '2024-01-01', '2024-06-30', 50000.00, 'active'),
        ('Mobile App Development', 'Create iOS and Android apps for customer engagement', '2024-02-01', '2024-12-31', 150000.00, 'active'),
        ('Data Migration', 'Migrate legacy systems to new cloud infrastructure', '2024-01-15', '2024-04-15', 75000.00, 'active'),
        ('Marketing Campaign', 'Q2 marketing campaign for new product launch', '2024-04-01', '2024-06-30', 25000.00, 'planning')
        ON CONFLICT (id) DO NOTHING;
        """
        
        insert_result = execute_tool(insert_projects_sql)
        if insert_result.get("success"):
            print("✓ Successfully inserted project data")
        else:
            print(f"❌ Insert failed: {insert_result.get('error', 'Unknown error')}")
        
        # Example 4: Query employees by department
        print("\n4. Finding engineering employees...")
        find_sql = """
        SELECT name, email, salary, hire_date 
        FROM employees 
        WHERE department = 'Engineering' 
        ORDER BY salary DESC
        """
        
        find_result = execute_tool(find_sql)
        if find_result.get("success"):
            engineers = find_result.get("data", [])
            print(f"✓ Found {len(engineers)} engineering employees:")
            for emp in engineers:
                name = emp.get('name', 'Unknown')
                email = emp.get('email', 'Unknown')
                salary = emp.get('salary', 0)
                hire_date = emp.get('hire_date', 'Unknown')
                # Handle salary formatting safely
                if isinstance(salary, (int, float)):
                    salary_str = f"${salary:,.2f}"
                else:
                    salary_str = str(salary)
                print(f"  - {name} ({email}): {salary_str}, Hired: {hire_date}")
        else:
            print(f"❌ Find failed: {find_result.get('error', 'Unknown error')}")
        
        # Example 5: Update employee salaries
        print("\n5. Updating employee salaries (5% raise for engineering)...")
        update_sql = """
        UPDATE employees 
        SET salary = salary * 1.05 
        WHERE department = 'Engineering'
        """
        
        update_result = execute_tool(update_sql)
        if update_result.get("success"):
            updated_count = update_result.get("data", {}).get("rowcount", 0)
            print(f"✓ Updated {updated_count} engineering employees with 5% raise")
        else:
            print(f"❌ Update failed: {update_result.get('error', 'Unknown error')}")
        
        # Example 6: Complex JOIN query
        print("\n6. Running complex JOIN query (employees and projects)...")
        
        join_sql = """
        SELECT 
            e.name as employee_name,
            e.department,
            p.name as project_name,
            p.status,
            p.budget
        FROM employees e
        CROSS JOIN projects p
        WHERE e.department = 'Engineering' AND p.status = 'active'
        ORDER BY p.budget DESC;
        """
        
        join_result = execute_tool(join_sql)
        if join_result.get("success"):
            results = join_result.get("data", [])
            print(f"✓ Found {len(results)} employee-project combinations:")
            
            for i, row in enumerate(results[:5]):  # Show first 5 results
                # Try different possible field names
                emp_name = (row.get('employee_name') or row.get('name') or 
                           row.get('e.name') or 'Unknown')
                dept = (row.get('department') or row.get('e.department') or 'Unknown')
                proj_name = (row.get('project_name') or row.get('p.name') or 'Unknown')
                budget = (row.get('budget') or row.get('p.budget') or 0)
                
                # Handle budget formatting safely
                if isinstance(budget, (int, float)):
                    budget_str = f"${budget:,.2f}"
                else:
                    budget_str = str(budget)
                print(f"  - {emp_name} ({dept}) → {proj_name}: {budget_str}")
        else:
            print(f"❌ JOIN query failed: {join_result.get('error', 'Unknown error')}")
        
        # Example 6b: JSON query format (new feature!)
        print("\n6b. Testing JSON query format...")
        json_query = '{"sql": "SELECT * FROM employees WHERE department = \'Engineering\'", "type": "select"}'
        
        json_result = execute_tool(json_query)
        if json_result.get("success"):
            results = json_result.get("data", [])
            print(f"✓ JSON query returned {len(results)} engineering employees:")
            for i, row in enumerate(results[:3]):
                print(f"  - {row.get('name', 'Unknown')} ({row.get('department', 'Unknown')})")
        else:
            print(f"❌ JSON query failed: {json_result.get('error', 'Unknown error')}")
        
        # Example 6c: Python object query format (new feature!)
        print("\n6c. Testing Python object query format...")
        python_query = {
            "sql": "SELECT name, department FROM employees WHERE salary > 50000",
            "type": "select"
        }
        
        python_result = execute_tool(python_query)
        if python_result.get("success"):
            results = python_result.get("data", [])
            print(f"✓ Python object query returned {len(results)} high-salary employees:")
            for i, row in enumerate(results[:3]):
                print(f"  - {row.get('name', 'Unknown')} ({row.get('department', 'Unknown')})")
        else:
            print(f"❌ Python object query failed: {python_result.get('error', 'Unknown error')}")
        
        # Example 7: Delete inactive projects
        print("\n7. Testing delete functionality...")
        delete_sql = "DELETE FROM projects WHERE status = 'planning'"
        
        delete_result = execute_tool(delete_sql)
        if delete_result.get("success"):
            deleted_count = delete_result.get("data", {}).get("rowcount", 0)
            print(f"✓ Deleted {deleted_count} planning projects")
        else:
            print(f"❌ Delete failed: {delete_result.get('error', 'Unknown error')}")
        
        # Example 8: Get database information
        print("\n8. Getting database information...")
        info_sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        
        info_result = execute_tool(info_sql)
        if info_result.get("success"):
            tables_data = info_result.get("data", [])
            print(f"✓ Database info:")
            print(f"  - Database: PostgreSQL (file-based mode)")
            
            # Safely handle tables display
            if isinstance(tables_data, list) and tables_data:
                table_names = [row.get('table_name', 'Unknown') for row in tables_data]
                print(f"  - Tables: {', '.join(table_names)}")
                print(f"  - Total tables: {len(table_names)}")
            else:
                print("  - Tables: None")
                print("  - Total tables: 0")
        else:
            print(f"❌ Info failed: {info_result.get('error', 'Unknown error')}")
        
        print("\n✓ PostgreSQL examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running PostgreSQL examples: {str(e)}")


def run_faiss_examples():
    """Run examples using FaissToolkit for vector database operations."""
    print("\n===== FAISS TOOLKIT EXAMPLES =====\n")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("To test FAISS examples, set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-openai-api-key-here'")
        print("Get your key from: https://platform.openai.com/api-keys")
        return
    
    try:
        # Initialize FAISS toolkit with default storage
        toolkit = FaissToolkit(
            name="DemoFaissToolkit",
            default_corpus_id="demo_corpus"
        )
        
        print("✓ FaissToolkit initialized with default storage")
        print(f"✓ Using OpenAI API key: {os.getenv('OPENAI_API_KEY')[:8]}...")
        
        # Get available tools
        insert_tool = toolkit.get_tool("faiss_insert")
        query_tool = toolkit.get_tool("faiss_query")
        list_tool = toolkit.get_tool("faiss_list")
        stats_tool = toolkit.get_tool("faiss_stats")
        delete_tool = toolkit.get_tool("faiss_delete")
        
        print(f"✓ Available tools: {[tool.name for tool in toolkit.get_tools()]}")
        
        # Example 1: Insert AI knowledge documents
        print("\n1. Inserting AI knowledge documents...")
        ai_documents = [
            "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "Deep learning is a specialized form of machine learning that uses neural networks with multiple layers to analyze and learn from data.",
            "Natural Language Processing (NLP) helps computers understand, interpret, and generate human language in a useful way.",
            "Computer vision enables machines to interpret and understand visual information from the world, including images and videos.",
            "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward.",
            "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.",
            "Transfer learning allows a model trained on one task to be adapted for a related task, improving efficiency and performance.",
            "Generative AI models can create new content, such as text, images, music, and code, based on patterns learned from training data.",
            "Explainable AI focuses on making AI systems' decisions and processes transparent and understandable to humans."
        ]
        
        insert_result = insert_tool(
            documents=ai_documents,
            metadata={
                "source": "ai_knowledge_base",
                "topic": "artificial_intelligence",
                "language": "en",
                "difficulty": "intermediate"
            }
        )
        
        if insert_result.get("success"):
            docs_inserted = insert_result.get("data", {}).get("documents_inserted", 0)
            chunks_created = insert_result.get("data", {}).get("chunks_created", 0)
            print(f"✓ Successfully inserted {docs_inserted} documents")
            print(f"  Chunks created: {chunks_created}")
        else:
            print(f"❌ Insert failed: {insert_result.get('error', 'Unknown error')}")
            return
        
        # Example 2: Perform semantic search queries
        print("\n2. Performing semantic search queries...")
        
        search_queries = [
            "How do machines learn?",
            "What is neural network?",
            "Explain deep learning",
            "How does AI generate content?",
            "What is computer vision?"
        ]
        
        for i, query in enumerate(search_queries, 1):
            print(f"\n  Query {i}: '{query}'")
            search_result = query_tool(
                query=query,
                top_k=3,
                similarity_threshold=0.1
            )
            
            if search_result.get("success"):
                results = search_result.get("data", {}).get("results", [])
                print(f"    ✓ Found {len(results)} relevant results:")
                for j, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    content = result.get('content', '')[:80]
                    print(f"      {j}. Score: {score:.3f} - {content}...")
            else:
                print(f"    ❌ Search failed: {search_result.get('error', 'Unknown error')}")
        
        # Example 3: Search with metadata filters
        print("\n3. Searching with metadata filters...")
        filtered_search_result = query_tool(
            query="machine learning algorithms",
            top_k=5,
            similarity_threshold=0.1,
            metadata_filters={"topic": "artificial_intelligence", "difficulty": "intermediate"}
        )
        
        if filtered_search_result.get("success"):
            results = filtered_search_result.get("data", {}).get("results", [])
            print(f"✓ Found {len(results)} results with metadata filters:")
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                content = result.get('content', '')[:100]
                metadata = result.get('metadata', {})
                print(f"  {i}. Score: {score:.3f} - {content}...")
                print(f"     Metadata: {metadata}")
        else:
            print(f"❌ Filtered search failed: {filtered_search_result.get('error', 'Unknown error')}")
        
        # Example 4: Get database statistics
        print("\n4. Getting database statistics...")
        stats_result = stats_tool()
        
        if stats_result.get("success"):
            stats = stats_result.get("data", {})
            print(f"✓ Database statistics:")
            print(f"  - Total corpora: {stats.get('total_corpora', 'Unknown')}")
            print(f"  - Corpora: {', '.join(stats.get('corpora', []))}")
            print(f"  - Embedding model: {stats.get('embedding_model', 'Unknown')}")
            print(f"  - Vector store type: {stats.get('vector_store_type', 'Unknown')}")
        else:
            print(f"❌ Stats failed: {stats_result.get('error', 'Unknown error')}")
        
        # Example 5: List all corpora
        print("\n5. Listing all corpora...")
        list_result = list_tool()
        
        if list_result.get("success"):
            corpora = list_result.get("data", {}).get("corpora", [])
            print(f"✓ Found {len(corpora)} corpora:")
            for corpus in corpora:
                corpus_id = corpus.get('corpus_id', 'Unknown')
                doc_count = corpus.get('document_count', 'Unknown')
                chunk_count = corpus.get('chunk_count', 'Unknown')
                print(f"  - {corpus_id}: {doc_count} documents, {chunk_count} chunks")
        else:
            print(f"❌ List failed: {list_result.get('error', 'Unknown error')}")
        
        # Example 6: Test delete functionality
        print("\n6. Testing delete functionality...")
        delete_result = delete_tool(
            metadata_filters={"source": "ai_knowledge_base"}
        )
        
        if delete_result.get("success"):
            deleted_count = delete_result.get("data", {}).get("deleted_count", 0)
            print(f"✓ Deleted {deleted_count} documents with metadata filter")
            
            # Verify deletion
            verify_result = query_tool(
                query="artificial intelligence",
                top_k=5,
                similarity_threshold=0.1
            )
            
            if verify_result.get("success"):
                remaining = verify_result.get('data', {}).get('total_results', 0)
                print(f"✓ Remaining documents after deletion: {remaining}")
        else:
            print(f"❌ Delete failed: {delete_result.get('error', 'Unknown error')}")
        
        print("\n✓ FAISS examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running FAISS examples: {str(e)}")
        if "DocumentMetadata" in str(e):
            print("Note: This appears to be a dependency issue with the RAG engine components")
            print("The FAISS toolkit may need additional setup or dependencies")


def main():
    """Main function to run all database tool examples."""
    print("===== DATABASE TOOLS EXAMPLES =====\n")
    
    # Run MongoDB examples
    run_mongodb_examples()
    
    # Run PostgreSQL examples
    run_postgresql_examples()
    
    # Run FAISS examples
    run_faiss_examples()
    
    print("\n===== ALL DATABASE EXAMPLES COMPLETED =====")


if __name__ == "__main__":
    main()
