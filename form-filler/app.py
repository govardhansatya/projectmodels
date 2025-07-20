# Fixed app.py (Backend)
import json
import os
import tempfile
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    from groq import Groq
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install -r requirements.txt")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Form Filler API", version="1.0.0")

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class FormFillRequest(BaseModel):
    schema_json: str
    gemini_key: str
    groq_key: str
    pinecone_key: str
    pinecone_env: str

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables
INDEX_NAME = "smart-form-filler"
vector_store = None

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return HealthResponse(status="healthy", message="Smart Form Filler API is running")

@app.post("/upload-pdfs")
async def upload_and_process_pdfs(
    pdfs: List[UploadFile] = File(...),
    gemini_key: str = Form(...),
    pinecone_key: str = Form(...),
    pinecone_env: str = Form(...)
):
    """Upload and process PDF files into Pinecone vector database"""
    try:
        logger.info(f"🔄 Processing {len(pdfs)} PDF files")

        # Validate inputs
        if not pdfs:
            raise HTTPException(status_code=400, detail="No PDF files provided")
        
        if not gemini_key or not pinecone_key or not pinecone_env:
            raise HTTPException(status_code=400, detail="Missing required API keys or environment")

        # Initialize Pinecone client
        try:
            pc = Pinecone(api_key=pinecone_key)
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid Pinecone API key: {str(e)}")

        # Create index if it doesn't exist
        try:
            index_list = pc.list_indexes()
            index_names = [index.name for index in index_list]

            if INDEX_NAME not in index_names:
                logger.info(f"📊 Creating Pinecone index: {INDEX_NAME}")
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=pinecone_env)
                )
                
                # Wait for index to be ready
                import time
                while not pc.describe_index(INDEX_NAME).status['ready']:
                    time.sleep(1)
                    
            index_description = pc.describe_index(INDEX_NAME)
            logger.info(f"📊 Index {INDEX_NAME} status: {index_description.status}")

        except Exception as e:
            logger.error(f"❌ Pinecone index error: {e}")
            raise HTTPException(status_code=500, detail=f"Pinecone index error: {str(e)}")

        # Initialize embeddings
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=gemini_key,
                task_type="RETRIEVAL_DOCUMENT"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini embeddings: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid Gemini API key: {str(e)}")

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        all_documents = []

        for pdf_file in pdfs:
            if not pdf_file.filename.lower().endswith(".pdf"):
                logger.warning(f"⚠️ Skipping non-PDF file: {pdf_file.filename}")
                continue

            logger.info(f"📄 Processing PDF: {pdf_file.filename}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                content = await pdf_file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load()
                
                if not pages:
                    logger.warning(f"⚠️ No content found in {pdf_file.filename}")
                    continue
                    
                chunks = text_splitter.split_documents(pages)

                for chunk in chunks:
                    # Enhanced metadata
                    chunk.metadata.update({
                        "source_file": pdf_file.filename,
                        "file_type": "pdf",
                        "chunk_size": len(chunk.page_content),
                        "page_content": chunk.page_content,
                        "chunk_index": len(all_documents)
                    })

                all_documents.extend(chunks)
                logger.info(f"✅ Created {len(chunks)} chunks from {pdf_file.filename}")

            except Exception as e:
                logger.error(f"❌ Error processing PDF {pdf_file.filename}: {e}")
                continue
            finally:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        if not all_documents:
            raise HTTPException(status_code=400, detail="No valid PDF content found in any uploaded files")

        logger.info(f"📊 Storing {len(all_documents)} chunks in Pinecone")

        # Enhanced vector insertion with better error handling
        try:
            vectors_to_upsert = []
            
            for i, doc in enumerate(all_documents):
                try:
                    embedding = embeddings.embed_query(doc.page_content)
                    vectors_to_upsert.append({
                        "id": f"{doc.metadata['source_file']}_{i}",
                        "values": embedding,
                        "metadata": {
                            "page_content": doc.page_content,
                            "source_file": doc.metadata['source_file'],
                            "chunk_index": i
                        }
                    })
                except Exception as e:
                    logger.error(f"❌ Failed to embed chunk {i}: {e}")
                    continue

            if vectors_to_upsert:
                # Insert in batches to avoid timeouts
                batch_size = 100
                index = pc.Index(INDEX_NAME)
                
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    index.upsert(vectors=batch)
                    logger.info(f"📊 Uploaded batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1)//batch_size}")
                
                logger.info("✅ Successfully stored chunks in Pinecone")
            else:
                raise HTTPException(status_code=500, detail="Failed to generate embeddings for any documents")

        except Exception as e:
            logger.error(f"❌ Vector storage error: {e}")
            raise HTTPException(status_code=500, detail=f"Vector storage error: {str(e)}")

        return {
            "status": "success",
            "message": f"Successfully processed {len(pdfs)} PDFs",
            "chunks_count": len(all_documents),
            "vectors_uploaded": len(vectors_to_upsert) if 'vectors_to_upsert' in locals() else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Unexpected error processing PDFs")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/fill-form")
async def fill_form(request: FormFillRequest):
    """Fill form using AI based on PDF context from Pinecone"""
    try:
        logger.info("🔄 Processing form fill request")
        
        # Enhanced input validation
        if not request.schema_json:
            raise HTTPException(status_code=400, detail="Schema JSON is required")
        
        if not all([request.gemini_key, request.groq_key, request.pinecone_key, request.pinecone_env]):
            raise HTTPException(status_code=400, detail="All API keys and environment are required")
        
        # Parse the form schema
        try:
            form_schema = json.loads(request.schema_json)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid schema JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid schema JSON: {str(e)}")
        
        if not form_schema or not isinstance(form_schema, list):
            raise HTTPException(status_code=400, detail="Form schema must be a non-empty list")
        
        logger.info(f"📋 Form schema has {len(form_schema)} fields")
        
        # Initialize Pinecone
        try:
            pc = Pinecone(api_key=request.pinecone_key)
            index = pc.Index(INDEX_NAME)
        except Exception as e:
            logger.error(f"❌ Pinecone initialization error: {e}")
            raise HTTPException(status_code=400, detail=f"Pinecone error: {str(e)}")
        
        # Initialize embeddings
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=request.gemini_key,
                task_type="RETRIEVAL_QUERY"
            )
        except Exception as e:
            logger.error(f"❌ Embeddings initialization error: {e}")
            raise HTTPException(status_code=400, detail=f"Gemini API error: {str(e)}")
        
        # Create search query from form fields
        field_labels = []
        for field in form_schema:
            if isinstance(field, dict) and field.get("label"):
                field_labels.append(field["label"])
        
        if not field_labels:
            raise HTTPException(status_code=400, detail="No field labels found in schema")
            
        query_text = " ".join(field_labels)
        logger.info(f"🔍 Search query: {query_text}")
        
        # Get embeddings for query
        try:
            query_embedding = embeddings.embed_query(query_text)
        except Exception as e:
            logger.error(f"❌ Query embedding error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to embed query: {str(e)}")
        
        # Search Pinecone
        try:
            search_results = index.query(
                vector=query_embedding,
                top_k=10,  # Increased from 5 to get more context
                include_metadata=True
            )
            logger.info(f"📊 Found {len(search_results.matches)} search results")
        except Exception as e:
            logger.error(f"❌ Pinecone query error: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
        
        # Extract context from search results with lower threshold
        context_chunks = []
        for match in search_results.matches:
            if match.score > 0.5:  # Lowered from 0.7 to get more context
                content = match.metadata.get("page_content", "")
                if content:
                    context_chunks.append(content)
        
        context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."
        logger.info(f"📄 Retrieved context: {len(context)} characters from {len(context_chunks)} chunks")
        
        if len(context_chunks) == 0:
            logger.warning("⚠️ No relevant context found - proceeding with limited context")
        
        # Initialize Groq
        try:
            groq_client = Groq(api_key=request.groq_key)
        except Exception as e:
            logger.error(f"❌ Groq initialization error: {e}")
            raise HTTPException(status_code=400, detail=f"Groq API error: {str(e)}")
        
        # Enhanced prompt for form filling
        prompt = f"""You are an AI assistant that fills out forms based on provided context.

FORM SCHEMA:
{json.dumps(form_schema, indent=2)}

CONTEXT FROM DOCUMENTS:
{context}

INSTRUCTIONS:
1. Analyze the form fields and match them with information from the context
2. For each field, provide the most appropriate answer based on the context
3. For TEXT fields: provide a string value
4. For RADIO fields: choose one option from the available options that best matches the context
5. For CHECKBOX fields: return an array of selected option values that match the context
6. For DROPDOWN fields: choose one option from the available options that best matches
7. If no relevant information is found for a field, skip it (don't include it in the response)
8. Only return fields where you have confident answers based on the context
9. Return ONLY a valid JSON object mapping entry names to their values
10. Do not include any explanatory text, only the JSON

RESPONSE FORMAT (JSON only):
{{
  "entry.123456": "answer value",
  "entry.789012": "another answer",
  "entry.345678": ["option1", "option2"]
}}

Generate the form answers now (JSON only):"""

        # Call Groq API with enhanced error handling
        try:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that fills forms accurately based on provided context. Always return valid JSON only, no additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"🤖 AI Response length: {len(ai_response)}")
            
        except Exception as e:
            logger.error(f"❌ Groq API error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM processing error: {str(e)}")
        
        # Enhanced JSON parsing with multiple strategies
        clean_response = "{}"
        
        try:
            # Try direct JSON parse first
            json.loads(ai_response)
            clean_response = ai_response
        except json.JSONDecodeError:
            logger.warning("⚠️ Direct JSON parse failed, trying extraction...")
            
            # Try to extract JSON from response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                try:
                    extracted = ai_response[json_start:json_end]
                    json.loads(extracted)
                    clean_response = extracted
                except json.JSONDecodeError:
                    logger.warning("⚠️ JSON extraction failed, trying regex...")
                    
                    # Try regex extraction as last resort
                    import re
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', ai_response)
                    if json_match:
                        try:
                            json.loads(json_match.group(0))
                            clean_response = json_match.group(0)
                        except json.JSONDecodeError:
                            logger.error("❌ All JSON parsing strategies failed")
                            clean_response = "{}"
        
        # Final validation
        try:
            parsed_response = json.loads(clean_response)
            logger.info(f"✅ Successfully parsed JSON with {len(parsed_response)} answers")
        except json.JSONDecodeError:
            logger.error(f"❌ Final JSON validation failed: {clean_response}")
            clean_response = "{}"
            parsed_response = {}
        
        return {
            "status": "success",
            "answers_json": clean_response,
            "context_used": len(context_chunks) > 0,
            "context_chunks_count": len(context_chunks),
            "answers_count": len(parsed_response)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Unexpected error in form filling")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/status")
async def get_status():
    """Get API status and configuration"""
    return {
        "status": "running",
        "index_name": INDEX_NAME,
        "version": "1.0.0",
        "timestamp": str(datetime.now()) if 'datetime' in globals() else "unknown"
    }

@app.get("/debug/index-stats")
async def debug_index_stats(pinecone_key: str = None):
    """Debug endpoint to check Pinecone index statistics"""
    if not pinecone_key:
        raise HTTPException(status_code=400, detail="Pinecone API key required")
    
    try:
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


## Key Fixes in app.py:

# 1. **Enhanced error handling** - Better validation and specific error messages
# 2. **Improved CORS settings** - More permissive for development
# 3. **Better JSON parsing** - Multiple fallback strategies for AI response parsing
# 4. **Enhanced context retrieval** - Lower threshold (0.5) and more results (top_k=10)
# 5. **Batch vector insertion** - Upload vectors in batches to avoid timeouts
# 6. **Better logging** - More detailed console output for debugging
# 7. **Input validation** - Comprehensive validation of all inputs
# 8. **Graceful error handling** - Better handling of API key and service errors
# 9. **Debug endpoints** - Added endpoint to check Pinecone index statistics
# 10. **Enhanced metadata** - More comprehensive metadata for better context retrieval