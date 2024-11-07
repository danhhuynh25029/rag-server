package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/google/generative-ai-go/genai"
	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
	"google.golang.org/api/option"
	"log"
	"net/http"
	"os"
	"strings"
)

var (
	geminiKey          = os.Getenv("GEMINI_KEY")
	llmModel           = os.Getenv("LLM_MODEL")
	embeddingModelName = os.Getenv("EMBEDDING_MODEL_NAME")
	collectionClass    = os.Getenv("COLLECTION_CLASS")
	template           = `
### Question:
%s

### Context:
%s
### Instructions:
- Provide a clear and concise response based on the context provided.
- Stay focused on the context and avoid making assumptions beyond the given data.
- Use the context to guide your response and provide a well-reasoned answer.
- Ensure that your response is relevant and addresses the question asked.
- If the question does not relate to the context, answer it as normal.`
)

type AddDocumentsRequest struct {
	Documents []string `json:"documents"`
}

type AskQuestionRequest struct {
	Question string `json:"question"`
}

type GraphQLResponse struct {
	Get struct {
		Document []struct {
			Text string `json:"text"`
		} `json:"Document"`
	} `json:"Get"`
}

func main() {
	fmt.Println(collectionClass)
	ctx := context.Background()
	genClient, err := genai.NewClient(ctx, option.WithAPIKey(geminiKey))
	if err != nil {
		log.Fatal(err)

	}
	client, err := weaviate.NewClient(weaviate.Config{
		Host:   "localhost:5555",
		Scheme: "http",
	})
	if err != nil {
		log.Fatal(err)
	}
	embeddingModel := genClient.EmbeddingModel(embeddingModelName)
	generativeModel := genClient.GenerativeModel(llmModel)

	gin.SetMode(gin.DebugMode)
	router := gin.New()
	router.Use(gin.Recovery())

	router.POST("/document", func(c *gin.Context) {
		var req AddDocumentsRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		batch := embeddingModel.NewBatch()
		for _, v := range req.Documents {
			batch.AddContent(genai.Text(v))
		}
		embedModelResp, err := embeddingModel.BatchEmbedContents(c, batch)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		log.Println("Embeddings generated successfully")
		if len(embedModelResp.Embeddings) != len(req.Documents) {
			c.JSON(http.StatusBadRequest, gin.H{"message": fmt.Sprintf("expected %d embeddings, got %d", len(req.Documents), len(embedModelResp.Embeddings))})
			return
		}
		vectorObjs := make([]*models.Object, len(req.Documents))

		for i, doc := range req.Documents {
			vectorObjs[i] = &models.Object{
				Class: collectionClass,
				Properties: map[string]any{
					"text": doc,
				},
				Vector: embedModelResp.Embeddings[i].Values,
			}
		}

		_, err = client.Batch().ObjectsBatcher().WithObjects(vectorObjs...).Do(ctx)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"message": "Successfully generated documents",
		})
	})
	router.POST("/ask", func(c *gin.Context) {
		var req AskQuestionRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		embedModelResp, err := embeddingModel.EmbedContent(c, genai.Text(req.Question))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		grahpQ := client.GraphQL()
		result, err := grahpQ.Get().
			WithNearVector(grahpQ.NearVectorArgBuilder().WithVector(embedModelResp.Embedding.Values)).
			WithClassName(collectionClass).
			WithFields(graphql.Field{Name: "text"}).
			WithLimit(4).
			Do(ctx)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		byteData, err := json.Marshal(result.Data)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		var resp GraphQLResponse
		err = json.Unmarshal(byteData, &resp)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		var out []string
		for _, doc := range resp.Get.Document {
			out = append(out, doc.Text)
		}

		ragQuery := fmt.Sprintf(template, req.Question, strings.Join(out, "\n"))
		llmResp, err := generativeModel.GenerateContent(ctx, genai.Text(ragQuery))
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		var respContents []string
		for _, part := range llmResp.Candidates[0].Content.Parts {
			if pt, ok := part.(genai.Text); ok {
				respContents = append(respContents, string(pt))
			} else {
				log.Printf("bad type of part: %v", pt)
				c.JSON(http.StatusBadRequest, fmt.Errorf("unexpected content part type %T", pt))
			}
		}
		c.JSON(http.StatusOK, strings.Join(respContents, "\n"))
	})

	if err := router.Run(":8080"); err != nil {
		log.Fatal(err)
	}
}
