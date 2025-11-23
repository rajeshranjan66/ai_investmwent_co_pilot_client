import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_chroma import Chroma  # Updated import
import argparse

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def ingest_pdf_to_chroma(pdf_path, persist_directory="chroma_db", chunk_size=1000, chunk_overlap=200):
    """
    Enhanced PDF ingestion with better text splitting and error handling

    Args:
        pdf_path: Path to the PDF file
        persist_directory: Directory to store ChromaDB
        chunk_size: Size of text chunks (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
    """
    try:
        # 1. Load PDF with error handling
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if not docs:
            logging.warning(f"No content found in PDF: {pdf_path}")
            return False

        # 2. Improved text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Better for academic papers
        )

        splits = text_splitter.split_documents(docs)

        logging.info(f"Split {len(docs)} pages into {len(splits)} chunks")

        # 3. Add metadata for better retrieval
        for i, split in enumerate(splits):
            if not split.metadata:
                split.metadata = {}
            split.metadata.update({
                "source": os.path.basename(pdf_path),
                "chunk_id": i,
                "total_chunks": len(splits)
            })

        # 4. Create embeddings and store - using updated imports
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=100  # Better for large documents
        )

        # Check if ChromaDB already exists and update instead of overwriting
        if os.path.exists(persist_directory):
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            # Add new documents to existing collection
            db.add_documents(splits)
            logging.info(f"Added documents to existing ChromaDB at '{persist_directory}'")
        else:
            # Create new collection
            db = Chroma.from_documents(
                splits,
                embeddings,
                persist_directory=persist_directory
            )
            logging.info(f"Created new ChromaDB at '{persist_directory}'")

        # Chroma 0.4.x+ automatically persists, so no need for manual persist()
        # db.persist()  # Removed this line

        logging.info(f"PDF '{pdf_path}' successfully ingested. {len(splits)} chunks stored in ChromaDB.")
        return True

    except Exception as e:
        logging.error(f"Error ingesting PDF {pdf_path}: {str(e)}")
        return False


def ingest_multiple_pdfs(pdf_directory, persist_directory="chroma_db", chunk_size=1000, chunk_overlap=200):
    """
    Process multiple PDFs in a directory
    """
    if not os.path.exists(pdf_directory):
        logging.error(f"Directory not found: {pdf_directory}")
        return 0

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    if not pdf_files:
        logging.warning(f"No PDF files found in directory: {pdf_directory}")
        return 0

    logging.info(f"Found {len(pdf_files)} PDF files to process")

    successful_ingestions = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        logging.info(f"Processing: {pdf_file}")

        if ingest_pdf_to_chroma(pdf_path, persist_directory, chunk_size, chunk_overlap):
            successful_ingestions += 1
        else:
            logging.error(f"Failed to process: {pdf_file}")

    logging.info(f"Successfully processed {successful_ingestions}/{len(pdf_files)} PDFs")
    return successful_ingestions


def test_retrieval(query, persist_directory="chroma_db", k=3):
    """
    Test retrieval from ChromaDB
    """
    try:
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        results = db.similarity_search(query, k=k)

        print(f"\n🔍 Retrieval Test for query: '{query}'")
        print("=" * 60)

        for i, doc in enumerate(results):
            print(f"\n📄 Result {i + 1}:")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Content: {doc.page_content[:200]}...")
            print(f"   Metadata: {dict(list(doc.metadata.items())[:3])}")  # Show first 3 metadata items

        return results

    except Exception as e:
        print(f"Error testing retrieval: {e}")
        return None


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="PDF Ingestion Tool for ChromaDB")
    parser.add_argument("--pdf", help="Path to a single PDF file")
    parser.add_argument("--dir", help="Path to directory containing PDF files")
    parser.add_argument("--db", default="chroma_db", help="ChromaDB directory (default: chroma_db)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    parser.add_argument("--test-query", help="Test retrieval with a query after ingestion")

    args = parser.parse_args()

    print("📚 PDF to ChromaDB Ingestion Tool")
    print("=" * 40)

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY=your_key_here")
        return

    if args.pdf:
        # Process single PDF
        if not os.path.exists(args.pdf):
            print(f"❌ PDF file not found: {args.pdf}")
            return

        print(f"Processing single PDF: {args.pdf}")
        success = ingest_pdf_to_chroma(
            args.pdf,
            args.db,
            args.chunk_size,
            args.chunk_overlap
        )

        if success:
            print(f"✅ Successfully ingested: {args.pdf}")
        else:
            print(f"❌ Failed to ingest: {args.pdf}")

    elif args.dir:
        # Process directory of PDFs
        print(f"Processing directory: {args.dir}")
        count = ingest_multiple_pdfs(
            args.dir,
            args.db,
            args.chunk_size,
            args.chunk_overlap
        )
        print(f"✅ Processed {count} PDF files")

    else:
        print("ℹ️  No action specified. Use --pdf or --dir to process files.")
        print("\nUsage examples:")
        print("  python pdf_ingestion.py --pdf research_paper.pdf")
        print("  python pdf_ingestion.py --dir ./papers/ --db my_chroma_db")
        print("  python pdf_ingestion.py --pdf paper.pdf --test-query \"machine learning trends\"")
        return

    # Test retrieval if requested
    if args.test_query:
        print(f"\n🧪 Testing retrieval with query: '{args.test_query}'")
        test_retrieval(args.test_query, args.db)


if __name__ == "__main__":
    main()

# Single PDF ingestion example
# python pdf_ingestion_tool.py --pdf research_paper.pdf
# python utils/ingest_pdf_to_chroma.py --pdf /Users/rajeshranjan/RajeshWork/mcp/mcp_project/research_pdf/GPS_Report_Invest_USA.pdf --test-query "investment strategies in USA"

# Ingest all PDFs in a directory:
# python pdf_ingestion_tool.py --dir ./research_papers/ --db my_chroma_db
# python utils/ingest_pdf_to_chroma.py --dir ./research_pdf/ --chroma_db

# With custom chunking parameters
# python pdf_ingestion_tool.py --pdf paper.pdf --chunk-size 1500 --chunk-overlap 300

# Test retrieval after ingestion:
# python pdf_ingestion_tool.py --pdf paper.pdf --test-query "quantum computing applications"

# Complete example with all options:
# python pdf_ingestion_tool.py --dir ./papers/ --db research_db --chunk-size 1200 --chunk-overlap 250 --test-query "AI market trends"