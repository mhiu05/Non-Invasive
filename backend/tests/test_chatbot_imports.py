"""
Test chatbot imports and functionality.
Run: cd backend && python tests/test_chatbot_imports.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("=== Testing imports ===")

# 1. Check langchain versions
try:
    import langchain
    print(f"[OK] langchain: {langchain.__version__}")
except ImportError as e:
    print(f"[FAIL] langchain: {e}")

try:
    import langchain_core
    print(f"[OK] langchain_core: {langchain_core.__version__}")
except ImportError as e:
    print(f"[FAIL] langchain_core: {e}")

try:
    import langchain_community
    print(f"[OK] langchain_community: {langchain_community.__version__}")
except ImportError as e:
    print(f"[FAIL] langchain_community: {e}")

try:
    import langchain_google_genai
    print(f"[OK] langchain_google_genai: {getattr(langchain_google_genai, '__version__', 'installed')}")
except ImportError as e:
    print(f"[FAIL] langchain_google_genai: {e}")

try:
    import langchain_text_splitters
    print(f"[OK] langchain_text_splitters: {getattr(langchain_text_splitters, '__version__', 'installed')}")
except ImportError as e:
    print(f"[FAIL] langchain_text_splitters: {e}")

# 2. Try different import paths for create_retrieval_chain
print("\n=== Testing chain imports ===")

paths_retrieval = [
    "langchain.chains:create_retrieval_chain",
    "langchain.chains.retrieval:create_retrieval_chain",
    "langchain_classic.chains:create_retrieval_chain",
    "langchain_classic.chains.retrieval:create_retrieval_chain",
]

for path in paths_retrieval:
    module_path, attr = path.split(":")
    try:
        mod = __import__(module_path, fromlist=[attr])
        obj = getattr(mod, attr)
        print(f"[OK] from {module_path} import {attr}")
    except (ImportError, AttributeError) as e:
        print(f"[FAIL] from {module_path} import {attr} -> {e}")

paths_stuff = [
    "langchain.chains.combine_documents:create_stuff_documents_chain",
    "langchain_classic.chains.combine_documents:create_stuff_documents_chain",
]

for path in paths_stuff:
    module_path, attr = path.split(":")
    try:
        mod = __import__(module_path, fromlist=[attr])
        obj = getattr(mod, attr)
        print(f"[OK] from {module_path} import {attr}")
    except (ImportError, AttributeError) as e:
        print(f"[FAIL] from {module_path} import {attr} -> {e}")

# 3. Test other imports used in engine.py
print("\n=== Testing engine imports ===")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("[OK] ChatGoogleGenerativeAI")
except ImportError as e:
    print(f"[FAIL] ChatGoogleGenerativeAI: {e}")

try:
    from langchain_core.prompts import ChatPromptTemplate
    print("[OK] ChatPromptTemplate")
except ImportError as e:
    print(f"[FAIL] ChatPromptTemplate: {e}")

# 4. Test loader imports
print("\n=== Testing loader imports ===")

try:
    from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
    print("[OK] TextLoader, DirectoryLoader, PyPDFLoader")
except ImportError as e:
    print(f"[FAIL] loaders: {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("[OK] FAISS")
except ImportError as e:
    print(f"[FAIL] FAISS: {e}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("[OK] HuggingFaceEmbeddings")
except ImportError as e:
    print(f"[FAIL] HuggingFaceEmbeddings: {e}")

print("\n=== Done ===")
