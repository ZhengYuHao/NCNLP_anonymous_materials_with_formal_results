# DSL v1.2 语法规范

本规范在 DSL v1.1 基础上,吸收 SPL 的结构性原语,解决"多场景复杂任务表达力不足"问题。与 v1.1 完全向后兼容。

## 0. SEMANTIC_BLOCK 块(新增 — 用于 LLM 语义理解)

```
SEMANTIC_BLOCK <block_id> "<中文描述>"
    [ NAME "<显示名称>" ]
    MODEL <model_name>                    # gpt-4o / gpt-4o-mini 等
    [ TEMPERATURE <float> ]              # 可选，默认 0.1
    [ MAX_TOKENS <int> ]                 # 可选，默认 2048
    TASK <task_type>                      # classification | generation | scoring | similarity | extraction

    INPUT:
        <var_name>: <type>               # String | Integer | Float | Boolean | Map | BloggerInfo | OrderInfo | ...
        ...
    OUTPUT:
        <var_name>: <type>
        ...
    PROMPT "<prompt 内容>"
ENDSEMANTIC_BLOCK
```

### TASK 类型说明

| TASK | 用途 | Temperature 建议 | 输出格式 |
|------|------|-----------------|---------|
| classification | 意图分类 | 0.1-0.3 | `{ intent: String, confidence: Float }` |
| generation | 内容生成 | 0.5-0.7 | `{ result: String }` |
| scoring | 多维度评分 | 0.1-0.3 | `{ score_a: Float, score_b: Float, ... }` |
| similarity | 语义相似度 | 0.1 | `{ similarity: Float }` |
| extraction | 信息抽取 | 0.1-0.3 | `{ field: Value, ... }` |

### SEMANTIC_BLOCK 示例

```swift
SEMANTIC_BLOCK sb_analyze_match "商单匹配分析"
    MODEL gpt-4o
    TEMPERATURE 0.3
    TASK classification
    INPUT:
        blogger_info: BloggerInfo
        order_info: OrderInfo
    OUTPUT:
        is_match: Boolean
        confidence: Float
        dimension_scores: Map
    PROMPT "你是商单博主推荐分析师..."
ENDSEMANTIC_BLOCK
```

### CALL 语句扩展(支持调用 SEMANTIC_BLOCK)

在 BLOCK 内调用 SEMANTIC_BLOCK：

```
CALL <semantic_block_id>
    INPUT { <var>: {{<ctx_var>}}, ... }
    OUTPUT { <var>: {{<ctx_var>}}, ... }
```

**示例**：
```
CALL sb_analyze_match
    INPUT { blogger_info: {{blogger_info}}, order_info: {{order_info}} }
    OUTPUT { is_match: {{is_match}}, confidence: {{confidence}} }
```

## 1. 顶层结构

```
AGENT {{AgentName}} "agent description"

    PERSONA: ... ENDPERSONA                        # 必填
    [ CONSTRAINTS: ... ENDCONSTRAINTS ]            # 推荐
    [ INPUTS: ... ENDINPUTS ]                      # 推荐
    [ OUTPUTS: ... ENDOUTPUTS ]                    # 推荐

    [ DEFINE {{var}}: Type [= default] ]*          # v1.1 保留

    [ BLOCK id "desc" ... ENDBLOCK ]+              # 至少一个
    [ BLOCK b_fallback "兜底" ... ENDBLOCK ]       # 强烈建议

    [ EXAMPLES: ... ENDEXAMPLES ]                  # 推荐

ENDAGENT
```

## 2. PERSONA 块(新增)

```
PERSONA:
    ROLE: <role description>
    [ <CapabilityKey>: <capability description> ]*
ENDPERSONA
```

`ROLE` 行必须且仅出现一次。其余行以 UPPER_SNAKE_CASE 作为 Key。

## 3. CONSTRAINTS 块(新增)

```
CONSTRAINTS:
    [ <ConstraintKey>: <constraint description> ]+
ENDCONSTRAINTS
```

推荐的标准 Key:
- `OUTPUT` — 输出格式约束
- `PRIORITY` — 优先级规则
- `EXCLUSION` — 排除条件执行规则
- `SAFETY` — 安全/合规约束

## 4. INPUTS / OUTPUTS 块(新增)

```
INPUTS:
    [ ("REQUIRED" | "OPTIONAL") ] {{var_name}} : Type [ "=" default_value ]
    ...
ENDINPUTS

OUTPUTS:
    {{var_name}} : Type
    ...
ENDOUTPUTS
```

缺省 `REQUIRED/OPTIONAL` 时视为 `REQUIRED`。类型沿用 v1.1 的 `String/Integer/Float/Boolean/List/Dict/Any`。

## 5. BLOCK 块(新增 — 本版本最核心改动)

```
BLOCK <block_id> "block description"
    # 任意 v1.1 语句(DEFINE 除外) + 下述 CONTAINS 扩展
ENDBLOCK
```

**语义规则**:
- `block_id` 必须唯一,建议按 `b<priority>_<topic>` 命名(如 `b1_bargain`)
- 块按**书写顺序**执行——这就是优先级编码方式
- 块内允许 `RETURN`,执行到即跳出整个 Agent
- 块之间是**顺序(非并行)**关系,上一块未 RETURN 则进入下一块

## 6. CONTAINS 算子扩展(新增 — 本版本最重要改动)

v1.1 已有 `CONTAINS` 但仅支持标量。v1.2 扩展为:

```
<var_ref> CONTAINS [ <string_literal> { "," <string_literal> } ]
<var_ref> NOT CONTAINS [ <string_literal> { "," <string_literal> } ]
```

**语义**:
- `x CONTAINS [a, b, c]` ≡ `(a IN x) OR (b IN x) OR (c IN x)`(任一命中即真)
- `x NOT CONTAINS [a, b, c]` ≡ `NOT (x CONTAINS [a, b, c])`(全部未命中才真)

**示例**:
```
IF {{user_input}} CONTAINS ["返点", "价格", "预算"]
    ...
ENDIF

IF {{user_input}} NOT CONTAINS ["已付款", "催款"]
    ...
ENDIF
```

**与 v1.1 兼容性**: 当 `[]` 中只有一个元素时退化为标量匹配,等价于 v1.1 的 `CONTAINS`。

## 7. "包含 ∧ 排除" 双条件范式(强制)

业务文档每个场景都有"触发条件 / 排除条件"二元结构。DSL v1.2 强制使用嵌套 IF 表达:

```
BLOCK b_scene_X "场景 X"
    IF {{user_input}} CONTAINS [/* 触发关键词 */]
        IF {{user_input}} NOT CONTAINS [/* 排除关键词 */]
            {{scene}} = X
            {{agent}} = "..."
            RETURN {{scene}}, {{agent}}
        ENDIF
    ENDIF
ENDBLOCK
```

如果业务文档无排除条件,内层 IF 可省略。

## 8. 多子场景的处理范式

一个场景内部有多个互斥子场景(如催稿的"未按时交 / 超时 / 未按模板")时,使用 `ELIF`:

```
BLOCK b3_urge_draft "催稿"
    IF {{user_input}} CONTAINS [/* 子场景 1 关键词 */]
        ... RETURN 3
    ELIF {{user_input}} CONTAINS [/* 子场景 2 关键词 */]
        ... RETURN 3
    ELIF {{user_input}} CONTAINS [/* 子场景 3 关键词 */]
        ... RETURN 3
    ENDIF
ENDBLOCK
```

## 9. LLM + 规则混合调度模式(规范)

**禁止**:
```
# ❌ 空心化源头
{{scene}} = CALL classify_scene({{user_input}})
```

**允许且推荐**:
```
# ✅ 用 CALL 获取结构化中间结果,再用规则分流
{{intent}} = CALL extract_intent({{user_input}})
IF {{intent}}.polarity != "abandon"
    IF {{user_input}} CONTAINS [...]
        ...
    ENDIF
ENDIF
```

**判定标准**:

| 情形 | 必须使用 |
|---|---|
| 业务文档列出了明确关键词 | `IF CONTAINS [...]` |
| 需要整句语义推理(如"抱怨但不放弃") | `CALL` 拿结构化结果,再 `IF` 分流 |
| 上下文相关的阶段判断 | `CALL infer_stage(...)` + `IF {{stage}} ==` |

## 10. EXAMPLES 块(新增)

```
EXAMPLES:
    CASE:
        INPUT: { <key>: <value>, ... }
        EXPECTED: { <key>: <value>, ... }
        EXECUTION_PATH: [ <block_id> {, <block_id>} ]
    ENDCASE
    ...
ENDEXAMPLES
```

**作用**:
- 编译器可据此生成 pytest 单元测试
- 是防止 Python 代码空心化的最后一道保险:测试不过 = 实现不完整
- `EXECUTION_PATH` 引用的 `block_id` 必须在 BLOCK 中已定义

## 11. 完整 EBNF

```ebnf
agent          ::= "AGENT" var_ref string_lit
                   persona [constraints] [inputs] [outputs]
                   { define_stmt }
                   { semantic_block }
                   { block }
                   [ examples ]
                   "ENDAGENT" ;

semantic_block ::= "SEMANTIC_BLOCK" ident string_lit
                   "MODEL" ident
                   [ "TEMPERATURE" float ]
                   [ "MAX_TOKENS" integer ]
                   "TASK" ident
                   "INPUT:" { ident ":" type }
                   "OUTPUT:" { ident ":" type }
                   "PROMPT" string_lit
                   "ENDSEMANTIC_BLOCK" ;

block          ::= "BLOCK" ident string_lit { stmt } "ENDBLOCK" ;

persona        ::= "PERSONA:"
                   "ROLE:" description
                   { ident ":" description }
                   "ENDPERSONA" ;

constraints    ::= "CONSTRAINTS:"
                   { ident ":" description }
                   "ENDCONSTRAINTS" ;

inputs         ::= "INPUTS:"
                   { [ "REQUIRED" | "OPTIONAL" ] var_ref ":" type [ "=" value ] }
                   "ENDINPUTS" ;

outputs        ::= "OUTPUTS:"
                   { var_ref ":" type }
                   "ENDOUTPUTS" ;

define_stmt    ::= "DEFINE" var_ref ":" type [ "=" value ] ;

block          ::= "BLOCK" ident string_lit { stmt } "ENDBLOCK" ;

stmt           ::= assign | if_stmt | for_stmt | while_stmt
                 | call_stmt | return_stmt | comment ;

call_stmt      ::= "CALL" ident
                   "INPUT" "{" { ident ":" var_ref } "}"
                   "OUTPUT" "{" { ident ":" var_ref } "}"
                 | var_ref "=" "CALL" ident "(" [ arglist ] ")"
                   [ "=" "await" python_expr ] ;

if_stmt        ::= "IF" cond { stmt }
                   { "ELIF" cond { stmt } }
                   [ "ELSE" { stmt } ]
                   "ENDIF" ;

cond           ::= simple_cond
                 | contains_cond
                 | not_contains_cond
                 | cond "AND" cond
                 | cond "OR" cond
                 | "NOT" cond
                 | "(" cond ")" ;

contains_cond     ::= var_ref "CONTAINS" string_list ;
not_contains_cond ::= var_ref "NOT" "CONTAINS" string_list ;
string_list       ::= "[" string_lit { "," string_lit } "]" ;

return_stmt    ::= "RETURN" var_ref { "," var_ref } ;

examples       ::= "EXAMPLES:" { case } "ENDEXAMPLES" ;
case           ::= "CASE:"
                   "INPUT:" dict_lit
                   "EXPECTED:" dict_lit
                   "EXECUTION_PATH:" "[" ident { "," ident } "]"
                   "ENDCASE" ;

type           ::= "String" | "Integer" | "Float" | "Boolean"
                 | "List" | "Dict" | "Any" ;

var_ref        ::= "{{" ident "}}" ;
```

## 12. 编译器验证清单

编译器/校验器必须检查:

1. 所有 `BLOCK` 有配对 `ENDBLOCK`;`IF/ELIF/ELSE/ENDIF` 配对
2. 所有 `var_ref` 要么在 `INPUTS/OUTPUTS/DEFINE` 声明,要么是 "运行时变量"(允许动态使用)
3. `BLOCK` 的 `block_id` 全局唯一
4. `EXAMPLES` 的 `EXECUTION_PATH` 引用的 block 都存在
5. 每个 `BLOCK` 应在所有执行路径上至少能 `RETURN` 或流经下一块
6. `CALL` 若带 `= await` 子句,实现必须只引用已知对象(`vector_db`/`llm`/Python 内置等),禁止自创函数名
7. 至少存在一个兜底 BLOCK(常名为 `b_fallback`),无条件 `RETURN`
