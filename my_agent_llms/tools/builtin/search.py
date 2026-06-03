import os
from typing import Any, Dict, List, Optional

from my_agent_llms.tools.base import Tool, ToolParameter


class SearchTool(Tool):
    """
    智能混合搜索工具

    支持多种搜索引擎后端，智能选择最佳搜索源:
    1. 混合模式 (hybrid) - 智能选择TAVILY或SERPAPI
    2. Tavily API (tavily) - 专业AI搜索
    3. SerpApi (serpapi) - 传统Google搜索
    """

    side_effect_free = True  # 只读网络查询,无本地副作用 → 可并行

    def __init__(self, backend: str = "hybrid", tavily_key: Optional[str] = None, serpapi_key: Optional[str] = None):
        super().__init__(
            name="search",
            description="一个智能网页搜索引擎。支持混合搜索模式，自动选择最佳搜索源。"
        )
        self.backend = (backend or "hybrid").lower()
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.tavily_client = None
        self.available_backends: List[str] = []
        self._setup_backends()

    def _setup_backends(self):
        """根据已配置的 Key 与已安装的依赖，登记可用后端。"""
        if self.tavily_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
            except ImportError:
                print("⚠️ 未安装 tavily-python，跳过 Tavily 后端。可执行: pip install tavily-python")
            except Exception as e:
                print(f"⚠️ Tavily 初始化失败:{e}")

        if self.serpapi_key:
            try:
                # 仅验证依赖是否存在，真正请求时再 import
                import serpapi  # noqa: F401
                self.available_backends.append("serpapi")
            except ImportError:
                print("⚠️ 未安装 google-search-results，跳过 SerpApi 后端。可执行: pip install google-search-results")

        if not self.available_backends:
            print("⚠️ 当前无可用搜索后端。请配置 TAVILY_API_KEY 或 SERPAPI_API_KEY。")

    def _search_hybrid(self, query: str) -> str:
        """混合搜索 - 智能选择最佳搜索源"""
        # 优先使用Tavily（AI优化的搜索）
        if "tavily" in self.available_backends:
            try:
                return self._search_tavily(query)
            except Exception as e:
                print(f"⚠️ Tavily搜索失败: {e}")
                # 如果Tavily失败，尝试SerpApi
                if "serpapi" in self.available_backends:
                    print("🔄 切换到SerpApi搜索")
                    return self._search_serpapi(query)

        # 如果Tavily不可用，使用SerpApi
        elif "serpapi" in self.available_backends:
            try:
                return self._search_serpapi(query)
            except Exception as e:
                print(f"⚠️ SerpApi搜索失败: {e}")

        # 如果都不可用，提示用户配置API
        return "❌ 没有可用的搜索源，请配置TAVILY_API_KEY或SERPAPI_API_KEY环境变量"

    def _search_tavily(self, query: str) -> str:
        """使用Tavily搜索"""
        response = self.tavily_client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=3
        )

        result = f"🎯 Tavily AI搜索结果:{response.get('answer', '未找到直接答案')}\n\n"

        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:200]}...\n"
            result += f"    来源: {item.get('url', '')}\n\n"

        return result

    def _search_serpapi(self, query: str) -> str:
        """使用 SerpApi 进行 Google 搜索。"""
        from serpapi import GoogleSearch

        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": 5,
            "hl": "zh-cn",
        }
        response = GoogleSearch(params).get_dict()

        if "error" in response:
            return f"❌ SerpApi 返回错误:{response['error']}"

        organic = response.get("organic_results", []) or []
        if not organic:
            return "🔍 SerpApi 未返回任何结果"

        lines = ["🎯 SerpApi 搜索结果:\n"]
        for i, item in enumerate(organic[:5], 1):
            lines.append(f"[{i}] {item.get('title', '')}")
            snippet = (item.get("snippet") or "").strip()
            if snippet:
                lines.append(f"    {snippet[:200]}")
            lines.append(f"    来源: {item.get('link', '')}\n")

        return "\n".join(lines)

    def run(self, parameters: Dict[str, Any]) -> str:
        query = (
            parameters.get("query")
            or parameters.get("input")
            or parameters.get("q")
            or ""
        )
        query = str(query).strip()
        if not query:
            return "❌ 搜索关键词不能为空"

        try:
            if self.backend == "tavily":
                if "tavily" not in self.available_backends:
                    return "❌ Tavily 后端不可用，请配置 TAVILY_API_KEY 并安装 tavily-python"
                return self._search_tavily(query)

            if self.backend == "serpapi":
                if "serpapi" not in self.available_backends:
                    return "❌ SerpApi 后端不可用，请配置 SERPAPI_API_KEY 并安装 google-search-results"
                return self._search_serpapi(query)

            return self._search_hybrid(query)
        except Exception as e:
            return f"❌ 搜索失败:{e}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="待搜索的关键词或问题",
                required=True,
            ),
        ]
