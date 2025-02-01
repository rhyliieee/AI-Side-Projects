import os
from dotenv import load_dotenv
import logging
from time import perf_counter

from .helper_utils import structure_context

import numpy as np
import torch.nn.functional as F
import torch
import faiss
from sklearn.decomposition import PCA

import nomic
from nomic import atlas, embed, AtlasDataset
from nomic.data_inference import NomicEmbedOptions
from nomic.dataset import AtlasMapData

from sentence_transformers import SentenceTransformer

from typing import Tuple, Callable, Optional

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

# SET UP DATA LOADER
#class ContextLoader:
    # def __init__(self, query: str, dataset_name: str, nomic_token: str, hf_token: str):
    #     self.query = query
    #     self.nomic_token = nomic_token
    #     self.hf_token = hf_token
    #     self.dataset_name = dataset_name
    #     self.embed_model = None
    
    # def pull_context(self) -> Tuple[AtlasMapData, np.ndarray]:
    #     # LOAD DATASET FROM NOMIC ATLAS
    #     textual_data, data_embeddings = self.load_dataset(self.nomic_token, self.dataset_name)
        
    #     # EMBED QUERY
    #     _, dims = data_embeddings.shape
    #     query_embedding = self.embed_query(self.query, dimensions=dims)
        
    #     # RERANKING SCANNER
    #     reranker = self.pca_reranking_scanner(data_embeddings, use_pca=True)
    #     # OBTAIN TOP K INDICES
    #     top_k_indices = reranker(query_embedding)
        
    #     return (textual_data, top_k_indices)
    
    # def load_dataset(self, nomic_token: str, dataset_name: str) -> Tuple[AtlasMapData, np.ndarray]:
    #     # LOAD DATASET FROM NOMIC ATLAS
    #     atlas_dataset = AtlasDataset(nomic_token, dataset_name)
    #     atlas_projection = atlas_dataset.maps[0]
        
    #     # GET TEXT EMBEDDINGS FROM NOMIC ATLAS
    #     data_embeddings = atlas_projection.embeddings.latent
        
    #     # GET ATLAS TEXT DATA
    #     textual_data = atlas_projection.data
    #     return (textual_data, data_embeddings)
    
    # def embed_query(self, query: str, dimensions: int) -> np.ndarray:  
    #     # LOAD EMBEDDING MODEL
    #     embed_model = load_embedding_model(hf_token=self.hf_token)
        
    #     # PREFIX THE QUERY WITH "search_query:"
    #     prefixed_query = f"search_query: {query}"
        
    #     # GENERATE EMBEDDINGS
    #     embedding = embed_model.encode(
    #         [prefixed_query],
    #         convert_to_tensor=True,  # Outputs a PyTorch tensor
    #         normalize_embeddings=False,  # We'll handle normalization ourselves
    #     )
        
    #     # APPLY LAYER NORMALIZATION
    #     embedding = F.layer_norm(embedding, (embedding.shape[1],))
        
    #     # TRUNCATE TO THE FIRST `matryoshka_dim` DIMENSIONS
    #     matryoshka_dim = dimensions
    #     embedding = embedding[:, :matryoshka_dim]
        
    #     # L2-NORMALIZE THE EMBEDDINGS
    #     embedding = F.normalize(embedding, p=2, dim=1) # Important for cosine similarity
        
    #     # CONVERT TO NUMPY ARRAY AND RETURN THE FIRST (AND ONLY) EMBEDDING
    #     return embedding[0].cpu().numpy()
    
    # def pca_reranking_scanner(data_embeddings: np.ndarray, dimensions: int = 256, use_pca: bool = False) -> callable:
    #     # Normalize full embeddings for proper cosine similarity
    #     embeddings_normalized = data_embeddings / np.linalg.norm(data_embeddings, axis=1, keepdims=True)
        
    #     # Dimensionality reduction setup
    #     if use_pca:
    #         pca = PCA(n_components=dimensions)
    #         reduced_embeddings = pca.fit_transform(embeddings_normalized).astype('float32')
    #         faiss.normalize_L2(reduced_embeddings)
    #     else:
    #         reduced_embeddings = embeddings_normalized[:, :dimensions].astype('float32')
    #         faiss.normalize_L2(reduced_embeddings)
        
    #     # FAISS index for efficient first-pass retrieval
    #     index = faiss.IndexFlatIP(dimensions)
    #     index.add(reduced_embeddings)

    #     def rerank_knn(query, k=5, expand=10) -> np.ndarray:
    #         expanded_k = k * expand
            
    #         # Normalize query
    #         query_normalized = (query / np.linalg.norm(query)).astype('float32')
            
    #         # Dimensionality reduction for query
    #         if use_pca:
    #             query_reduced = pca.transform(query_normalized.reshape(1, -1)).astype('float32')
    #             faiss.normalize_L2(query_reduced)
    #         else:
    #             query_reduced = query_normalized[:dimensions].reshape(1, -1)
    #             faiss.normalize_L2(query_reduced)
            
    #         # First-pass retrieval using FAISS
    #         _, candidate_indices = index.search(query_reduced, expanded_k)
    #         candidate_indices = candidate_indices.flatten()
            
    #         # Second-pass reranking with full embeddings
    #         full_candidates = embeddings_normalized[candidate_indices]
    #         scores_second_pass = query_normalized @ full_candidates.T
    #         top_k_indices = np.argsort(-scores_second_pass)[:k]
            
    #         return candidate_indices[top_k_indices]
        
    #     return rerank_knn
    
    # def to_dict(self, atlas_map_data: AtlasMapData) -> dict:
    #     return atlas_map_data.tb.to_pylist()
    
class ContextLoader:
    """A pipeline for contextual query processing using Nomic Atlas embeddings and FAISS-based retrieval.
    
    Args:
        query: User query string
        dataset_name: Name of Nomic Atlas dataset
        nomic_token: API token for Nomic Atlas
        hf_token: HuggingFace token for model access
    """
    
    def __init__(self, query: str, dataset_name: str, nomic_token: str, hf_token: str):
        self.query = query
        self.nomic_token = nomic_token
        self.hf_token = hf_token
        self.dataset_name = dataset_name
        self._embed_model = None  # Lazy-loaded embedding model
        nomic.login(nomic_token)
        
    def pull_context(self, k: int = 5, expand: int = 10) -> Tuple[AtlasMapData, np.ndarray]:
        """Main execution flow for contextual query processing.
        
        Args:
            k: Number of top results to return
            expand: Expansion factor for initial candidate retrieval
            
        Returns:
            Tuple containing textual data and top indices
        """
        textual_data, data_embeddings = self._load_dataset()
        query_embedding = self._embed_query(data_embeddings)
        reranker = self._create_reranker(data_embeddings)
        return textual_data, reranker(query_embedding, k=k, expand=expand)
    
    def _load_dataset(self) -> Tuple[AtlasMapData, np.ndarray]:
        """Load and preprocess dataset from Nomic Atlas."""
        atlas_dataset = AtlasDataset(identifier=self.dataset_name)
        if not atlas_dataset.maps:
            raise ValueError(f"No maps found in dataset {self.dataset_name}")
            
        atlas_projection = atlas_dataset.maps[0]
        data_embeddings = atlas_projection.embeddings.latent.astype(np.float32)
        
        # Pre-normalize embeddings once for efficient similarity calculations
        data_embeddings /= np.linalg.norm(data_embeddings, axis=1, keepdims=True)
        return atlas_projection.data, data_embeddings
    
    def _embed_query(self, data_embeddings: np.ndarray) -> np.ndarray:
        """Generate and process query embeddings with dimension validation."""
        if self._embed_model is None:
            self._embed_model = self._load_embedding_model()
        
        prefixed_query = f"search_query: {self.query}"
        embedding = self._embed_model.encode(
            [prefixed_query],
            convert_to_tensor=True,
            normalize_embeddings=False,
        )
        
        return self._process_embedding(embedding, data_embeddings.shape[1]).cpu().numpy()
    
    def _process_embedding(self, embedding: torch.Tensor, dataset_dims: int) -> torch.Tensor:
        """Process embeddings through normalization and dimension validation."""
        embedding = F.layer_norm(embedding, (embedding.shape[1],))
        
        if embedding.shape[1] < dataset_dims:
            raise ValueError(
                f"Embedding dim {embedding.shape[1]} < dataset dim {dataset_dims}. "
                "Consider reducing PCA dimensions."
            )
            
        embedding = embedding[:, :dataset_dims]
        return F.normalize(embedding, p=2, dim=1)
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """Load and validate the sentence transformer model."""
        
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            token=self.hf_token
        )
        
        if not hasattr(model, 'encode'):
            raise RuntimeError("Invalid model architecture - missing encode method")
        return model
    
    def _create_reranker(self, data_embeddings: np.ndarray) -> Callable:
        """Factory method for creating reranker instances."""
        return Reranker.create(
            original_embeddings=data_embeddings,
            use_pca=True
        )

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Reranker:
    """Efficient two-stage retrieval system using FAISS and PCA dimensionality reduction.
    
    Args:
        index: FAISS index for approximate search
        original_embeddings: Normalized original embeddings for final scoring
        pca: Optional PCA model for dimensionality reduction
    """
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, index: faiss.Index, original_embeddings: np.ndarray, pca: Optional[PCA] = None):
        start_time = perf_counter()
        self.logger.info("INITIALIZING RERANKER INSTANCE")
        
        self.index = index
        self.original_embeddings = original_embeddings
        self.pca = pca
        
        self.logger.debug(f"INITIALIZED WITH INDEX TYPE: {type(index)}, "
                         f"EMBEDDINGS SHAPE: {original_embeddings.shape}, "
                         f"PCA: {'enabled' if pca else 'disabled'}")
        self.logger.info(f"RERANKER INITIALIZATION COMPLETED IN {perf_counter() - start_time:.2f}s")
        
    @classmethod
    def create(cls, original_embeddings: np.ndarray, use_pca: bool) -> "Reranker":
        """Factory method for creating configured reranker instances."""
        start_time = perf_counter()
        cls.logger.info("Creating new Reranker instance")

        processed_embeddings, pca = cls._prepare_embeddings(original_embeddings, use_pca)
        
        index = faiss.IndexFlatIP(processed_embeddings.shape[1])
        index.add(processed_embeddings)
        
        cls.logger.info(f"FAISS INDEX CREATED WITH {index.ntotal} VECTORS")
        cls.logger.info(f"RERANKER CREATION COMPLEMED IN {perf_counter() - start_time:.2f}s")

        return cls(index, original_embeddings, pca)
    
    @classmethod
    def _prepare_embeddings(cls, embeddings: np.ndarray, use_pca: bool) -> Tuple[np.ndarray, Optional[PCA]]:
        """Prepare embeddings for FAISS indexing with optional PCA."""
        start_time = perf_counter()
        cls.logger.info("PREPARING EMBEDDINGS FOR INDEXING")
        cls.logger.debug(f"ORIGINAL EMBEDDINGS SHAPE: {embeddings.shape}")


        if use_pca:
            cls.logger.info("APPLYING PCA FOR DIMENSIONALITY REDUCTION")

            pca = PCA(n_components=min(256, embeddings.shape[1]))
            processed = pca.fit_transform(embeddings).astype(np.float32)
            faiss.normalize_L2(processed)
            cls.logger.debug(f"REDUCED EMBEDDINGS SHAPE: {processed.shape}")

            return processed, pca
            
        # Direct truncation without PCA
        processed = embeddings[:, :256].astype(np.float32)
        faiss.normalize_L2(processed)
        
        cls.logger.info(f"EMBEDDINGS PREPARATION COMPLETED IN {perf_counter() - start_time:.2f}s")

        return processed, None
    
    def __call__(self, query: np.ndarray, k: int = 5, expand: int = 10) -> np.ndarray:
        """Execute the two-stage retrieval process.
        
        Args:
            query: Query embedding vector
            k: Final number of results to return
            expand: Multiplier for initial candidate pool
            
        Returns:
            Array of indices for top matches
        """
        processed_query = self._prepare_query(query)
        return self._two_pass_search(orig_query=query, processed_query=processed_query, k=k, expand=expand)
    
    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        """Normalize and transform query for index compatibility."""
        query = query.astype(np.float32)
        query /= np.linalg.norm(query)
        
        if self.pca:
            query = self.pca.transform(query.reshape(1, -1))
        else:
            query = query[:256].reshape(1, -1)
            
        faiss.normalize_L2(query)
        return query
    
    def _two_pass_search(self, orig_query: np.ndarray, processed_query: np.ndarray,
                         k: int, expand: int) -> np.ndarray:
        """Two-phase retrieval with approximate search and exact reranking."""
        # BROAD INITIAL SEARCH USING REDUCED DIMENSIONS
        _, candidates = self.index.search(processed_query, k * expand)
        candidates = candidates.flatten()
        
        # PRECISE RERANKING USING ORIGINAL EMBEDDINGS
        candidate_embeddings = self.original_embeddings[candidates]
        scores = orig_query.reshape(1, -1) @ candidate_embeddings.T # ENSURE CORRECT DIMENSIONS
        return candidates[np.argsort(-scores)[0, :k]]

if __name__ == "__main__":
    # Load environment variables
    nomic_token = os.getenv("DOHCPG_TOKEN")
    hf_token=os.getenv("HF_TOKEN")
    
    sample_user_prompt = "How should modifiable lifestyle risk factors like smoking and obesity be addressed in the periodic health examination for cardiovascular disease prevention?"
    
    # Initialize context loader
    loader = ContextLoader(sample_user_prompt, "DOHCPG-DOCS", nomic_token, hf_token)
    text_data, top_indices = loader.pull_context()
    # CONVERT TEXT DATA TO PYARRAW TABLE
    text_data_arrow = text_data.tb
    
    context = structure_context(text_data_arrow, top_indices)
    print(f"{'*' * 50}\n\n{context}\n\n{'*' * 50}")