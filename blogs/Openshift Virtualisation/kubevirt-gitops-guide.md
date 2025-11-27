# Modernizing Virtual Machine Management with OpenShift Virtualization and GitOps

## Introduction

The cloud-native revolution has transformed how we build and deploy applications, with containers becoming the preferred packaging format. However, many organizations still have legacy applications running on virtual machines that aren't ready for containerization. This is where Openshift Virtualization comes in - a technology that brings virtual machine management to Kubernetes.

In this blog post, we'll explore how to implement a Openshift Virtualization infrastructure using GitOps principles, providing a modern approach to VM management that aligns with cloud-native practices.

## What is Openshift Virtualization?

Red Hat® OpenShift® Virtualization, an included feature of Red Hat OpenShift, provides a modern platform for organizations to run and deploy their new and existing virtual machine (VM) workloads. The solution allows for easy migration and management of traditional virtual machines onto a trusted, consistent, and comprehensive hybrid cloud application platform.

OpenShift Virtualization simplifies the migration of your VMs while offering a path for infrastructure modernization, taking advantage of the simplicity and speed of a cloud-native application platform. It aims to preserve existing virtualization investments while embracing modern management principles, and it’s the foundation for Red Hat’s comprehensive virtualization solution.

Key benefits of OpenShift Virtualization include:
- Running VMs alongside containers in the same cluster
- Using Kubernetes tools to manage VM workloads
- Simplifying infrastructure by consolidating management platforms
- Enabling gradual migration from VMs to containers

More info [here](https://www.redhat.com/en/engage/15-reasons-adopt-openshift-virtualization-ebook)

## GitOps: The Foundation for Modern Infrastructure Management

GitOps applies DevOps best practices to infrastructure automation, using Git as the single source of truth. With GitOps:

1. Your desired infrastructure state is declared in Git
2. Automated processes ensure the actual state matches the desired state
3. Changes follow a clear workflow: commit, review, approve, and deploy
4. The entire history of your infrastructure is versioned and auditable

# 🚀 OpenShift Virtualization GitOps Automation - How to


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![OpenShift](https://img.shields.io/badge/OpenShift-4.10+-red.svg)](https://www.openshift.com/)
[![ArgoCD](https://img.shields.io/badge/ArgoCD-Powered-green.svg)](https://argoproj.github.io/cd/)


**A GitOps approach to managing OpenShift Virtualization virtualization on OpenShift clusters**


---

## 📋 Overview

This [GIT repository](https://github.com/rmallam/kubevirt-gitops)  contains automation scripts and configuration for setting up a GitOps workflow using Argo CD (OpenShift GitOps) to manage OpenShift Virtualization resources. The project provides a declarative approach to managing virtual machines and related resources on OpenShift.

## ✅ Prerequisites

- An OpenShift cluster 4.10+ with cluster-admin access
- `oc` command-line tool installed and configured
- `virtctl` command-line tool installed (optional)
- Basic understanding of GitOps principles and Argo CD

## 🚦 Quick Start
### Openshift login (OpenShift GitOps)
```bash
 oc login -u username -p password openshiftapi 
```

### Install Argo CD (OpenShift GitOps)

Run the installation script:

```bash
# Make the script executable
chmod +x install-argo.sh

# Run the script
./install-argo.sh
```
This script will install openshift gitops operator and output the details of the argocd URL, username and password as shown below.

 ```bash
 $./install-argo.sh
[2025-03-28 14:15:02] Retrieving ArgoCD access information...

=== ArgoCD Access Information ===
ArgoCD URL: https://openshift-gitops-server-openshift-gitops.test.openshiftapps.com
Username: admin
Password: dummy
```

![argocd](argo-dashboard.png)
## 🔒 How to use private git repo with ArgoCD?

If your repo is private, argocd will need access to pull the code from your repo. Run the following command to add the GitHub token to ArgoCD.

[How to get github token?](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

```bash
SERVER_URL=$(oc get routes openshift-gitops-server -n openshift-gitops -o jsonpath='{.status.ingress[0].host}')
ADMIN_PASSWD=$(oc get secret openshift-gitops-cluster -n openshift-gitops -o jsonpath='{.data.admin\.password}' | base64 -d)
argocd login --username admin --password ${ADMIN_PASSWD} ${SERVER_URL} --grpc-web
argocd repo add https://github.com/rmallam/OpenShift Virtualization-gitops --username rmallam --password gitpat #replace this with the right git token
```

## 📦 Install Openshift virtualisation Operator

In this section, We will be installing the operators in the git-ops way. The helm chart [here](https://github.com/rmallam/kubevirt-gitops/tree/main/infrastructure/operators) will be used to deploy all the required operators, based on the definition in the values file.

the list of subscriptions in the values file with enabled: true will be deployed in the cluster. 

```yaml
channel: stable
version: 4.15.8
subscriptions:
  - name: advanced-cluster-management
    namespace: open-cluster-management
    channel: release-2.10
    source: redhat-operators
    enabled: false
  - name: kubevirt-hyperconverged
    namespace: openshift-cnv
    channel: stable
    source: redhat-operators
    enabled: true
```
This can either be deployed manually running in the command from the infrastructure folder  
```helm install operators ./operators```
or using argocd application. Apply the infra-appset.yaml to deploy operators

```bash
oc apply -f argo-apps/infra-appset.yaml
```
This will create an argo application called operators and deploys the helm chart in the cluster.

![argocd](argo-operator-application.png)
## 📝 Additional Notes

### ROSA (Red Hat OpenShift Service on AWS)

Adding a bar metal machine pool for ROSA to run virtualization workloads:

```bash
rosa create machinepools -c $(rosa list clusters | awk -F " " '{print $2}' | grep -v NAME) --instance-type m5.metal --name virt-pool --replicas 3
```
We have now finished installing all the required components to run a VM on openshift. lets deploy the VM's now.

## 🔄  Deploying the virtual machine.

We created a helm chart to deploy virtual machines easily into the openshift clusters. Refer [here](https://github.com/rmallam/kubevirt-gitops/tree/main/vm-examples/helm/fedora-vm) to review the helm chart.

We will be using the argo application set to deploy this Helm chart which will deploy the virtual machine on openshift.

```bash
oc apply -f vm-examples/argo-apps/helm-appset.yaml

applicationset.argoproj.io/virtualmachines-stretch-application-set created
```
This will create an argo app and deploy the vm into openshift.

![fedora-vm-progressing](fedora-vm-progressing.png)

As you can see in the image above, All the resources required for the virtual machine to run are being deployed. The status moves from progressing to Healthy in Argo once the changes have been completed deployed.

![fedora-vm-healthy](fedora-vm-healthy.png)

 use the oc cli to check if the virtual machine is running.

```bash
$oc get vm
NAME        AGE     STATUS    READY
fedora-vm   4m32s   Running   True
```

To login to the virtual machine, You can either use the virtctl or the load balancer service that gets created as part of this deployment.

To configure different users access to this VM, update the cloud-init.yaml secret with the required users and passwords.

```bash
virtctl console fedora-vm
Successfully connected to fedora-vm console. The escape sequence is ^]

logger-vm-primary-site login: rakesh
Password:
Last login: Fri Mar 28 05:25:03 on ttyS0
[rakesh@logger-vm-primary-site ~]$
```

```bash
ssh rakesh@a1bc5c65622b14596b526f7eca317280-730837299.us-east-1.elb.amazonaws.com
Warning: Permanently added 'a1bc5c65622b14596b526f7eca317280-730837299.us-east-1.elb.amazonaws.com' (ED25519) to the list of known hosts.
Last login: Fri Mar 28 05:26:05 2025
[rakesh@logger-vm-primary-site ~]$
```
## Control the state of VM using GITOPS

To control the state of the VM, Update the runStrategy value in values-primary.yaml file to the desired value and push the changes to git repo. if the desired state is different to the one from the cluster, Argocd will sync the changes and the VM will move the desired state.

lets change the runStrategy from Always which is current to "Halted" to stop the VM.

```yaml
hostname:  logger-vm-primary-site

# Node placement configuration
# To use nodeSelector, uncomment and adjust the following:
# nodeSelector: 
#   #beta.kubernetes.io/instance-type: c6i.metal
#   site: primary
# 
# To disable nodeSelector completely, use an empty map:
#nodeSelector: {}

runStrategy: Halted #options: Halted, RerunOnFailure, Always, Manual
``` 

Once this is done, Argocd will sync the changes and the VM will stop.

![fedora-vm-stopped](fedora-vm-argo-stop.png)

```bash
$oc get vm
NAME        AGE   STATUS    READY
fedora-vm   21m   Stopped   False
```

```yaml
$oc get vm fedora-vm -o yaml
apiVersion: kubevirt.io/v1
kind: VirtualMachine
...
spec:
  dataVolumeTemplates:
  - metadata:
      creationTimestamp: null
      name: fedora-vm
    spec:
      preallocation: false
      source:
        http:
          url: https://download.fedoraproject.org/pub/fedora/linux/releases/40/Cloud/x86_64/images/Fedora-Cloud-Base-UEFI-UKI.x86_64-40-1.14.qcow2
      storage:
        resources:
          requests:
            storage: 30Gi
        storageClassName: gp3-csi
  runStrategy: **Halted**
  template:
  ....
```
Update the runStrategy back to Always in Values-common.yaml and push it to git repo to start the VM.

```yaml
hostname:  logger-vm-primary-site

# Node placement configuration
# To use nodeSelector, uncomment and adjust the following:
# nodeSelector: 
#   #beta.kubernetes.io/instance-type: c6i.metal
#   site: primary
# 
# To disable nodeSelector completely, use an empty map:
#nodeSelector: {}

runStrategy: Always #options: Halted, RerunOnFailure, Always, Manual
``` 
![fedora-vm-stopped](fedora-vm-argo-start.png)

```bash
oc get vm
NAME        AGE   STATUS    READY
fedora-vm   28m   Running   True

```

## Conclusion:

Red Hat OpenShift Virtualization offers a unified, scalable platform for migrating traditional virtual machines to Openshift which offer an common platform for both containers and VM's. It ensures consistent hybrid management and supports modernization efforts, enabling organizations to efficiently manage and deploy VM and container workloads with a comprehensive set of development and operations tools. It integrates seamlessly with existing tools like OpenShift GitOps, allowing for efficient management of workloads. 

Using GITOPS to manage the workloads will help in reducing any manual errors and helps to track the changes easily.