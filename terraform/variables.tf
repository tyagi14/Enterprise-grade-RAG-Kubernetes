variable "project_id" {
  description = "The GCP Project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for services"
  type        = string
  default     = "us-central1"
}

variable "app_name" {
  description = "Base name for all resources"
  type        = string
  default     = "enterprise-rag"
}

variable "qdrant_url" {
  description = "Qdrant Cloud endpoint"
  type        = string
}

variable "qdrant_api_key" {
  description = "Qdrant Cloud API Key"
  type        = string
  sensitive   = true
}

variable "groq_api_key" {
  description = "Groq API Key"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Postgres Password"
  type        = string
  sensitive   = true
}

variable "logfire_token" {
  description = "Logfire Write Token"
  type        = string
  sensitive   = true
}

variable "langsmith_api_key" {
  type      = string
  sensitive = true
}

variable "langsmith_project" {
  type    = string
  default = "rag_scale_test"
}

variable "doc_ai_processor_id" {
  description = "The Google Cloud Document AI Processor ID"
  type        = string
}

variable "groq_fallback_api_key" {
  description = "Groq fallback API key — used by Portkey on primary failure"
  type        = string
  sensitive   = true
}

variable "portkey_api_key" {
  description = "Portkey Gateway API key"
  type        = string
  sensitive   = true
}

variable "portkey_config_id" {
  description = "Portkey config ID for the LLM gateway config"
  type        = string
}

variable "nvidia_api_key" {
  description = "NVIDIA API key for NeMo Guardrails"
  type        = string
  sensitive   = true
}

variable "judge_groq" {
  description = "Separate Groq key used only for RAGAS eval judging (never exhausts prod key)"
  type        = string
  sensitive   = true
}

variable "judge_groq_2" {
  description = "Optional second Groq key for the RAGAS judge. Only adds daily budget if it belongs to a DIFFERENT Groq organisation (limits are per organisation). Empty = single-key judge."
  type        = string
  sensitive   = true
  default     = ""
}

variable "judge_groq_3" {
  description = "Optional third Groq key for the RAGAS judge (adds daily budget only if it belongs to a DIFFERENT Groq organisation than the other judge keys). Empty = unused."
  type        = string
  sensitive   = true
  default     = ""
}
