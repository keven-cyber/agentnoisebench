TOOLS = [
    # {
    #     "type": "function",
    #     "name": "search_documents",
    #     "description": "This is an online search service that can retrieve relevant document fragments based on user questions.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "query": {
    #                 "type": "string",
    #                 "description": "The user's question or query that will be directly used for online retrieval.",
    #             },
    #             "topk": {
    #                 "type": "integer",
    #                 "description": "The desired number of documents to return (1-10).",
    #                 "default": 3,
    #                 "minimum": 1,
    #                 "maximum": 10,
    #             },
    #         },
    #         "required": ["query"],
    #         "additionalProperties": False,
    #     },
    #     "strict": True,
    # }
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "This is an online search service that can retrieve relevant document fragments based on user questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question or query that will be directly used for online retrieval.",
                    },
                    "topk": {
                        "type": "integer",
                        "description": "The desired number of documents to return (1-10).",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    }
]