import asyncio
import json
from typing import List, Dict, Any

from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult,TextContent

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState


class WebMcpSearchNode(BaseNode):

    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        # 1.从state中 rewritten_query,item_names
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", [])

        # 2. 调用MCP工具进行网络搜索
        mcp_search_results = asyncio.run(self._call_mcp(rewritten_query))

        # 3.处理结果
        if mcp_search_results:
            return {"web_search_docs":mcp_search_results}
        return {"web_search_docs":[]}


    async def _call_mcp(self, rewritten_query)->List[Dict[str,Any]]:
        mcp_search_results = []
        # 创建MCP服务访问客户端
        async with MCPServerStreamableHttp(
            name="联网搜索",
            params={
                "url":self.config.mcp_dashscope_base_url,
                "headers":{"Authorization": f"Bearer {self.config.mcp_dashscope_api_key}"},
                "timeout": 60
            },
            cache_tools_list=True,
            max_retry_attempts=3
        ) as mcp_client:
            # 调用MCP服务下的工具
            mcp_response : CallToolResult = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": rewritten_query,
                    "count": 5,
                    "timeout": 60
                }
            )
            # 处理结果
            result: List[TextContent] = mcp_response.content
            # 从结果中获取JSON字符串
            json_str:str = result[0].text
            # 反序列化：str----obj
            obj = json.loads(json_str)  # {}
            pages =  obj.get("pages",[])
            for page in pages:
                title = page.get("title", "")
                url = page.get("url", "")
                snippet = page.get("snippet", "")
                mcp_search_results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })

        return mcp_search_results


if __name__ == '__main__':
    node = WebMcpSearchNode()
    state = {
        "rewritten_query": "华为擎云W585和华为显示器 B3-243H的参数",
        "item_names": [
            "华为擎云W585 台式计算机",
            "华为 B3-243H 显示器"
        ]
    }

    state = node.process(state)

    json_str = json.dumps(state,indent=4,ensure_ascii=False)
    print(json_str)