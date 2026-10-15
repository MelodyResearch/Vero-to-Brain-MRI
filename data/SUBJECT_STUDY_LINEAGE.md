# Subject/study lineage status

No accessible, approved MRI images have been linked to QA in this execution. Consequently all local usable subject, study, volume and task counts are **zero**. The SGMRI-VQA public `volume_id` is not proof of a unique patient. BrainTRACE public anonymous patient/study keys cannot supply its missing image files. NeuroQA QA paths are placeholders until source volumes are separately obtained and mapped.

Required local-only lineage table: `source_revision → source_subject_id → pseudonymous_group_key → study/visit date → series/volume UID → frame/slice index → QA ID`. Validate one-to-many mappings and same-patient visit order. Keep the table outside GitHub. Cross-source deduplication is mandatory before independent-source claims.
