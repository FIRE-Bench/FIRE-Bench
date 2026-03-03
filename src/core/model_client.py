"""
OpenAI API model client with direct configuration passing (thread-safe for multi-user environments)
"""

from json import load
import os
import asyncio
import random
from typing import Dict, Any, Optional, List, final
from loguru import logger
from tqdm import tqdm
import traceback

try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI 
except ImportError:
    logger.error("OpenAI library not found. Please install: pip install openai>=1.0.0")
    raise

from .base import BaseModelClient, BaseModelConfig

class OpenAIModelClient(BaseModelClient):
    """OpenAI API client with threading support and direct configuration"""
    
    def __init__(self, config: BaseModelConfig, max_workers: int = 128):
        self.config = config
        self.urls = config.urls # Use provided URLs or fallback to single URL
        self.max_workers = max_workers
        self._aclients: List[Any] = [] 
        self._cache_lock = asyncio.Lock()
        self._usage_lock = asyncio.Lock() 
        self._sem = asyncio.Semaphore(self.max_workers) 
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        
    async def __aenter__(self):
        # Initialize OpenAI clients for each URL
        api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
        api_type = self.config.api_type or os.getenv('OPENAI_API_TYPE')
        api_version = self.config.api_version or os.getenv('OPENAI_API_VERSION')
        
        if not api_key:
            raise ValueError("API key is required. Provide it in config or set OPENAI_API_KEY environment variable")
        
        # Create a client for each URL
        for base_url in self.urls:
            if api_type == "azure":
                client = AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=base_url,
                    max_retries=2
                )
            else:
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    max_retries=2
                )
            self._aclients.append(client)

        logger.info(f"Async OpenAI clients initialized for {len(self._aclients)} URLs: {self.urls}")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close the asynchronous clients one by one
        try:
            await asyncio.gather(*(c.close() for c in self._aclients), return_exceptions=True)
        finally:
            self._aclients.clear()
    
    def validate_config(self, config: BaseModelConfig) -> bool:
        """Validate configuration"""
        api_key = config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("API key is not provided in config or OPENAI_API_KEY environment variable")
            return False
        return True

    
    async def _make_async_request(self, prompt: str, progress_callback=None, **kwargs) -> str:   
        """Make asynchronous OpenAI API request"""
        try:
            stream_or_resp = None
            # Use configuration values, with kwargs overrides
            model = kwargs.get('model') or self.config.model
            temperature = float(kwargs.get('temperature', self.config.temperature))
            top_p = kwargs.get('top_p', self.config.top_p)
            top_k = kwargs.get('top_k', self.config.top_k)
            max_tokens = int(kwargs.get('max_tokens', self.config.max_tokens))
            system_prompt = kwargs.get('system_prompt', self.config.system_prompt)
            streaming = self.config.streaming
            use_chat = self.config.use_chat

            kkwargs: dict[str, Any] = {"extra_body":self.config.extra_body} 


            kkwargs["max_completion_tokens"] = max_tokens

            if not self._aclients:
                raise RuntimeError("No OpenAI clients initialized")
            client = random.choice(self._aclients)

            if top_p:
                kkwargs["top_p"] = top_p
            if top_k:
                kkwargs["extra_body"]["top_k"] = top_k
            if use_chat:
                messages = [{"role": "user", "content": prompt}]
                if system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})
                request_func = client.chat.completions.create
                kkwargs["messages"] = messages
            else:
                request_func = client.completions.create
                kkwargs["prompt"] = prompt
            
            # add repetition penalty for FIRE-RM
            if 'rm' in model.lower():
                kkwargs["extra_body"]["repetition_penalty"] = 1.05
                
            stream_or_resp = await request_func(
                model=model,
                temperature=temperature,
                timeout=float(self.config.timeout),
                stream=streaming,
                **kkwargs
            )
            if streaming:
                collected= []
                if use_chat:
                    content_name = "delta"
                else:
                    content_name = "text"
                async for chunk in stream_or_resp:
                    delta = getattr(chunk.choices[0], content_name, None)
                    piece = getattr(delta, "content", None) if delta else None
                    if piece:
                        collected.append(piece)
                        if progress_callback:
                            progress_callback({'type': 'content', 'content': piece, 'accumulated': ''.join(collected)})
                    if hasattr(chunk, "usage") and chunk.usage:
                        async with self._usage_lock:
                            self._total_input_tokens += chunk.usage.prompt_tokens
                            self._total_output_tokens += chunk.usage.completion_tokens
                final_content = ''.join(collected)
        
            else:
                if use_chat:
                    final_content = stream_or_resp.choices[0].message.content
                else:
                    final_content = stream_or_resp.choices[0].text
                async with self._usage_lock:
                    self._total_input_tokens += stream_or_resp.usage.prompt_tokens
                    self._total_output_tokens += stream_or_resp.usage.completion_tokens
                    
            if final_content is None:
                final_content = ""
            if progress_callback:
                if final_content:
                    progress_callback({'type': 'complete', 'content': final_content})
                else:
                    progress_callback({'type': 'error', 'error': final_content, 'error_message': stream_or_resp})
            return final_content

        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)} \n\n {traceback.format_exc()} \n\n{stream_or_resp}")
            return "" 
    
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts using thread pool with real-time progress tracking"""
        async with self._usage_lock:
            self._total_input_tokens = 0
            self._total_output_tokens = 0
        if not prompts:
            return []
        
        if not kwargs.get('cache_writer'):
            raise ValueError("cache_writer is required")
        cache_writer = kwargs['cache_writer']
        
        if not self._aclients:
            raise RuntimeError("No OpenAI clients initialized. Use async context manager.")
        
        logger.info(f"Processing batch of {len(prompts)} prompts with async concurrency={self.max_workers}")
        

        # results storage
        results: List[str] = [""] * len(prompts)
        streaming_status = {}  # track the streaming status of each request
        
        def create_progress_callback(index: int, prompt: str, pbar: tqdm):
            """Create a progress callback for each request"""
            def callback(data: Dict[str, str]):
                if data['type'] == 'content':
                    # update the streaming status display
                    streaming_status[index] = data.get('accumulated', '')[-10:].replace("\n", "")
                    # todo: add more detailed progress display here
                elif data['type'] == 'complete':
                    results[index] = data['content']
                    pbar.update(1)
                    # asynchronous write cache (avoid blocking the event loop)
                    payload = {"prompt": prompt, "model_response": results[index]}
                    async def _write():
                        async with self._cache_lock:
                            await asyncio.to_thread(cache_writer.write, payload)
                            if hasattr(cache_writer, "_fp"):
                                cache_writer._fp.flush()
                    asyncio.create_task(_write())
                    if index in streaming_status:
                        del streaming_status[index]
                elif data['type'] == 'error':
                    results[index] = ""
                    pbar.update(1)
                    logger.warning(f"Request {index} failed: {data['error']}\n error_message: {data['error_message']}\n\n")
                    if index in streaming_status:
                        del streaming_status[index]
            return callback

        async def run_one(index: int, prompt: str, pbar: tqdm):
            async with self._sem:
                """Asynchronously execute a single request"""
                progress_callback = create_progress_callback(index, prompt, pbar)
                return await self._make_async_request(prompt, progress_callback, **kwargs)
        
        pbar = tqdm(total=len(prompts), desc="Streaming Requests", unit="req", position=0, dynamic_ncols=True)

        tasks = [asyncio.create_task(run_one(i, p, pbar)) for i, p in enumerate(prompts)]
        # here we use as_completed to "harvest" the results in real-time, or we can directly await gather
        for coro in asyncio.as_completed(tasks):
            await coro
        logger.info(f"Total input tokens: {self._total_input_tokens}")
        logger.info(f"Total output tokens: {self._total_output_tokens}")
        return results
        


class ModelClientFactory:
    """Factory for creating model clients"""
    
    @staticmethod
    def create_client(config: BaseModelConfig) -> OpenAIModelClient:
        """Create OpenAI model client with direct configuration passing"""
        # No longer modifying global environment variables
        # Configuration is passed directly to the client
        
        # Extract multi-URL configuration if available
        urls = getattr(config, "urls")
        per_url_max_workers = getattr(config, 'per_url_max_workers')
        
        # Calculate total max_workers: number of URLs * per_url_max_workers
        total_max_workers = len(urls) * per_url_max_workers
        
        return OpenAIModelClient(config=config, max_workers=total_max_workers)